# Copyright (C) 2026 wuwukasi (wuwukaka)
# SPDX-License-Identifier: GPL-3.0-only

"""
Identity-based multi-person tracker for BodyRatioMapper.

Tracks people across video frames using nearest-neighbor matching with
sliding-window voting for robustness against position swaps and brief
occlusions.  Performs 65 % valid-rate filtering and t*-frame renumbering
after tracking completes.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class TrackState:
    """Mutable per-track state updated every frame."""
    track_id: int
    alive: bool = True
    # vote_window[fi] = candidate idx assigned at frame fi, or -1 (missing)
    vote_window: List[int] = field(default_factory=list)
    last_valid_ref_pt: Optional[np.ndarray] = None
    last_valid_frame_idx: int = -1
    missing_streak: int = 0
    # Per-frame output (single-person frame dicts)
    frames: List[dict] = field(default_factory=list)
    # assignment[fi] = candidate idx at frame fi, -1 = missing
    assignment: List[int] = field(default_factory=list)
    # Hysteresis state for _person_reference_point_with_mode
    ref_state: Dict = field(default_factory=lambda: {
        "ref_mode": "nose",
        "nose_miss_streak": 0,
        "nose_recover_streak": 0,
    })


@dataclass
class TrackingResult:
    """Output of MultiPersonTracker.track()."""
    tracks: List[List[dict]]   # n_alive_tracks x n_frames
    t_star: int                # first frame where all alive tracks coexist


# ---------------------------------------------------------------------------
# Helpers (standalone, no external deps beyond numpy)
# ---------------------------------------------------------------------------

def build_track_frame(person: dict, canvas_w: int, canvas_h: int) -> dict:
    return {"people": [person], "canvas_width": canvas_w, "canvas_height": canvas_h}


def align_tracks_to_n_ref(
    tracks: List[List[dict]],
    n_ref: int,
    zero_track_builder=None,
    clone_track_fn=None,
) -> Tuple[List[List[dict]], int, int, str]:
    """
    Trim or pad *tracks* in-place so len(tracks) == n_ref.

    Returns (tracks, trim_count, pad_count, pad_source).
    - trim_count > 0: excess tracks were removed (left-to-right kept).
    - pad_count  > 0: shortage was filled.
    - pad_source: "none" | "zero_track" | "last_valid_track"
    """
    trim_count = 0
    pad_count = 0
    pad_source = "none"

    if len(tracks) > n_ref:
        trim_count = len(tracks) - n_ref
        tracks[:] = tracks[:n_ref]
    elif len(tracks) < n_ref:
        pad_count = n_ref - len(tracks)
        if len(tracks) == 0:
            if zero_track_builder is not None:
                zt = zero_track_builder()
                tracks.extend([clone_track_fn(zt) if clone_track_fn else list(zt) for _ in range(n_ref)])
            pad_source = "zero_track"
        else:
            src = tracks[-1]
            while len(tracks) < n_ref:
                tracks.append(clone_track_fn(src) if clone_track_fn else list(src))
            pad_source = "last_valid_track"

    return tracks, trim_count, pad_count, pad_source


# ---------------------------------------------------------------------------
# Main tracker
# ---------------------------------------------------------------------------

class MultiPersonTracker:
    """
    Frame-by-frame multi-person tracker with sliding-window voting.

    Usage::

        tracker = MultiPersonTracker(conf_thresh=0.30, parent=node_instance)
        result = tracker.track(pose_keypoint_frames)
        candidate_tracks = result.tracks          # already filtered & renumbered
    """

    def __init__(
        self,
        parent,
        conf_thresh: float = 0.30,
        match_ratio: float = 0.08,
        vote_window_half: int = 7,
        max_missing_streak: int = 15,
        trajectory_pass_rate: float = 0.65,
        gap_mult_max: int = 3,
    ):
        self.parent = parent
        self.conf_thresh = conf_thresh
        self.match_ratio = match_ratio
        self.vote_window_half = vote_window_half
        self.max_missing_streak = max_missing_streak
        self.trajectory_pass_rate = trajectory_pass_rate
        self.gap_mult_max = gap_mult_max

        self.tracks: List[TrackState] = []
        self._next_id = 0
        self.n_frames = 0
        self._frames_raw: List[dict] = []   # original frames (for t*/renumber)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def track(self, frames: List[dict]) -> TrackingResult:
        """Main entry point.  *frames* is the raw POSE_KEYPOINT frame list."""
        self._frames_raw = frames
        self.n_frames = len(frames)
        if self.n_frames == 0:
            return TrackingResult(tracks=[], t_star=-1)

        # Phase A – sequential tracking with in-loop voting
        self._init_tracks_from_frame0(frames[0])
        for fi in range(1, self.n_frames):
            self._process_frame(fi, frames[fi])

        # Phase B – 65 % valid-rate filter
        alive = self._apply_trajectory_filter()

        # Phase C – t* (first frame where all alive tracks coexist)
        t_star = self._find_t_star(alive)

        # Phase D – renumber by X at t*
        if t_star >= 0:
            self._renumber_at_t_star(alive, t_star)

        # Phase E – ensure first frame valid
        self._ensure_first_frame_valid(alive)

        out_tracks = [t.frames for t in alive]
        print(f"[Tracker] n_frames={self.n_frames} raw_tracks={len(self.tracks)} "
              f"alive={len(alive)} t_star={t_star}")
        return TrackingResult(tracks=out_tracks, t_star=t_star)

    # ------------------------------------------------------------------
    # Initialisation (frame 0)
    # ------------------------------------------------------------------

    def _init_tracks_from_frame0(self, frame: dict) -> None:
        canvas_w = frame.get("canvas_width", 512)
        canvas_h = frame.get("canvas_height", 768)
        candidates = self._get_candidates(frame)

        for ci, person in enumerate(candidates):
            if self.parent._is_frame_absent(person, self.conf_thresh):
                continue
            if not self.parent._video_frame_passes_required_points(person, self.conf_thresh):
                continue
            ts = TrackState(track_id=self._next_id)
            self._next_id += 1
            ref_pt = self._ref_point(person, ts.ref_state)
            if ref_pt is None:
                # No usable reference point – create track but mark as missing frame-0
                ts.frames.append(build_track_frame(
                    self.parent._build_zero_person_openpose(), canvas_w, canvas_h))
                ts.assignment.append(-1)
                ts.vote_window.append(-1)
                ts.missing_streak = 1
            else:
                ts.last_valid_ref_pt = ref_pt
                ts.last_valid_frame_idx = 0
                ts.frames.append(build_track_frame(
                    self.parent._clone_person_fast(person), canvas_w, canvas_h))
                ts.assignment.append(ci)
                ts.vote_window.append(ci)
            self.tracks.append(ts)

    # ------------------------------------------------------------------
    # Per-frame matching
    # ------------------------------------------------------------------

    def _process_frame(self, fi: int, frame: dict) -> None:
        canvas_w = frame.get("canvas_width", 512)
        canvas_h = frame.get("canvas_height", 768)
        candidates = self._get_candidates(frame)
        n_cand = len(candidates)

        alive_tracks = [t for t in self.tracks if t.alive]
        if not alive_tracks:
            # All dead – still append empty frames for consistency
            for t in self.tracks:
                t.frames.append(build_track_frame(
                    self.parent._build_zero_person_openpose(), canvas_w, canvas_h))
                t.assignment.append(-1)
                t.vote_window.append(-1)
            return

        # --- Step 1: compute preference for each track ---
        # prefs[tid] = (chosen_cand_idx, distance, source)
        prefs: Dict[int, Tuple[int, float, str]] = {}

        # Pre-compute candidate reference points (each with its own state)
        cand_pts: List[Optional[np.ndarray]] = []
        for c in candidates:
            tmp_state = {"ref_mode": "nose", "nose_miss_streak": 0, "nose_recover_streak": 0}
            cand_pts.append(self._ref_point(c, tmp_state))

        for track in alive_tracks:
            nn_idx, nn_dist = self._nearest_candidate(track, cand_pts, canvas_w, canvas_h, fi)
            vote_pref = self._vote_majority(track, fi, n_cand)

            chosen_idx = nn_idx
            chosen_dist = nn_dist
            source = "nn"

            if vote_pref >= 0 and vote_pref != nn_idx and vote_pref < n_cand:
                d_vote = self._cand_distance(track.last_valid_ref_pt, cand_pts[vote_pref])
                # Per-track tau based on this track's gap
                gap = fi - track.last_valid_frame_idx if track.last_valid_frame_idx >= 0 else fi
                tau_track = self._match_threshold(canvas_w, canvas_h, gap)
                # Override with vote if: vote candidate is within tau,
                # AND either there's no nn or vote is at least as close.
                if d_vote <= tau_track and (nn_idx < 0 or d_vote <= nn_dist):
                    chosen_idx = vote_pref
                    chosen_dist = d_vote
                    source = "vote"

            prefs[track.track_id] = (chosen_idx, chosen_dist, source)

        # --- Step 2: conflict resolution (greedy by score) ---
        final_map = self._resolve_conflicts(prefs, alive_tracks, fi, cand_pts, canvas_w, canvas_h)

        # --- Step 3: detect new tracks from unmatched candidates ---
        assigned_cands = set(final_map.values())
        tau = self.parent._pixel_match_threshold(canvas_w, canvas_h, ratio=self.match_ratio)
        for ci in range(n_cand):
            if ci in assigned_cands:
                continue
            # Candidate not matched to any track – check if it's far from all tracks
            if self.parent._is_frame_absent(candidates[ci], self.conf_thresh):
                continue
            if not self.parent._video_frame_passes_required_points(candidates[ci], self.conf_thresh):
                continue
            pt = cand_pts[ci]
            if pt is None:
                continue
            too_close = False
            for t in alive_tracks:
                if t.last_valid_ref_pt is not None:
                    d = float(np.linalg.norm(t.last_valid_ref_pt - pt))
                    if d < tau:
                        too_close = True
                        break
            if not too_close:
                # Create new track
                ts = TrackState(track_id=self._next_id)
                self._next_id += 1
                # Pad previous frames with zeros
                for _ in range(fi):
                    ts.frames.append(build_track_frame(
                        self.parent._build_zero_person_openpose(), canvas_w, canvas_h))
                    ts.assignment.append(-1)
                    ts.vote_window.append(-1)
                ts.missing_streak = fi  # all previous frames were missing
                ref_pt = self._ref_point(candidates[ci], ts.ref_state)
                if ref_pt is not None:
                    ts.last_valid_ref_pt = ref_pt
                    ts.last_valid_frame_idx = fi
                ts.frames.append(build_track_frame(
                    self.parent._clone_person_fast(candidates[ci]), canvas_w, canvas_h))
                ts.assignment.append(ci)
                ts.vote_window.append(ci)
                if ts.missing_streak > self.max_missing_streak:
                    ts.alive = False
                self.tracks.append(ts)

        # --- Step 4: update state for existing tracks ---
        for track in alive_tracks:
            if track.track_id in final_map:
                ci = final_map[track.track_id]
                cand_person = candidates[ci]
                ref_pt = self._ref_point(cand_person, track.ref_state)
                if ref_pt is not None:
                    track.last_valid_ref_pt = ref_pt
                    track.last_valid_frame_idx = fi
                track.missing_streak = 0
                track.frames.append(build_track_frame(
                    self.parent._clone_person_fast(cand_person), canvas_w, canvas_h))
                track.assignment.append(ci)
                track.vote_window.append(ci)
            else:
                track.frames.append(build_track_frame(
                    self.parent._build_zero_person_openpose(), canvas_w, canvas_h))
                track.assignment.append(-1)
                track.vote_window.append(-1)
                track.missing_streak += 1
                if track.missing_streak > self.max_missing_streak:
                    track.alive = False

    # ------------------------------------------------------------------
    # Nearest candidate
    # ------------------------------------------------------------------

    def _nearest_candidate(
        self,
        track: TrackState,
        cand_pts: List[Optional[np.ndarray]],
        canvas_w: int,
        canvas_h: int,
        fi: int,
    ) -> Tuple[int, float]:
        """Return (best_cand_idx, distance).  Returns (-1, inf) if none."""
        if track.last_valid_ref_pt is None:
            return -1, float('inf')

        gap = fi - track.last_valid_frame_idx
        tau = self._match_threshold(canvas_w, canvas_h, gap)

        best_idx = -1
        best_dist = float('inf')
        for ci, pt in enumerate(cand_pts):
            if pt is None:
                continue
            d = float(np.linalg.norm(track.last_valid_ref_pt - pt))
            if d < best_dist:
                best_dist = d
                best_idx = ci

        # Single-track: always accept nearest (no threshold check)
        if len([t for t in self.tracks if t.alive]) <= 1:
            return best_idx, best_dist

        if best_dist > tau:
            return -1, float('inf')
        return best_idx, best_dist

    @staticmethod
    def _cand_distance(ref_pt: Optional[np.ndarray], cand_pt: Optional[np.ndarray]) -> float:
        if ref_pt is None or cand_pt is None:
            return float('inf')
        return float(np.linalg.norm(ref_pt - cand_pt))

    # ------------------------------------------------------------------
    # Sliding-window vote
    # ------------------------------------------------------------------

    def _vote_majority(self, track: TrackState, fi: int, n_candidates: int) -> int:
        """Return the majority candidate idx within the vote window, or -1."""
        half = self.vote_window_half
        vw = track.vote_window
        lo = max(0, fi - half)
        hi = min(len(vw), fi + half + 1)
        window = vw[lo:hi]
        valid = [c for c in window if 0 <= c < n_candidates]
        if not valid:
            return -1
        counts: Dict[int, int] = {}
        for c in valid:
            counts[c] = counts.get(c, 0) + 1
        return max(counts.items(), key=lambda x: x[1])[0]

    # ------------------------------------------------------------------
    # Conflict resolution
    # ------------------------------------------------------------------

    def _resolve_conflicts(
        self,
        prefs: Dict[int, Tuple[int, float, str]],
        alive_tracks: List[TrackState],
        fi: int,
        cand_pts: List[Optional[np.ndarray]],
        canvas_w: int,
        canvas_h: int,
    ) -> Dict[int, int]:
        """
        Greedy assignment: lower score wins.
        score = distance  (if chosen == vote_pref)
        score = distance + LARGE_PENALTY  (otherwise)
        Returns track_id -> final_cand_idx (only successfully assigned tracks).
        """
        VOTE_BONUS = 1e6

        # Build (score, track_id, cand_idx) tuples
        all_pairs = []
        for track in alive_tracks:
            if track.track_id not in prefs:
                continue
            cidx, cdist, source = prefs[track.track_id]
            if cidx < 0:
                continue
            # Reject if distance exceeds per-track tau
            gap = fi - track.last_valid_frame_idx if track.last_valid_frame_idx >= 0 else fi
            tau_track = self._match_threshold(canvas_w, canvas_h, gap)
            if cdist > tau_track:
                continue
            vote_pref = self._vote_majority(track, fi, len(cand_pts))
            score = cdist + (0 if cidx == vote_pref else VOTE_BONUS)
            all_pairs.append((score, track.track_id, cidx))

        all_pairs.sort(key=lambda x: x[0])

        final: Dict[int, int] = {}
        used_cands: set = set()

        for _score, tid, cidx in all_pairs:
            if tid in final or cidx in used_cands:
                continue
            final[tid] = cidx
            used_cands.add(cidx)

        # Fallback: unassigned tracks try their vote_majority directly
        for track in alive_tracks:
            if track.track_id in final:
                continue
            vp = self._vote_majority(track, fi, len(cand_pts))
            if vp >= 0 and vp not in used_cands:
                # Verify the vote candidate is within tau
                d_vp = self._cand_distance(track.last_valid_ref_pt, cand_pts[vp])
                gap = fi - track.last_valid_frame_idx if track.last_valid_frame_idx >= 0 else fi
                tau_track = self._match_threshold(canvas_w, canvas_h, gap)
                if d_vp <= tau_track:
                    final[track.track_id] = vp
                    used_cands.add(vp)

        return final

    # ------------------------------------------------------------------
    # Post-tracking: filter / t* / renumber / first-frame
    # ------------------------------------------------------------------

    def _apply_trajectory_filter(self) -> List[TrackState]:
        """Keep tracks whose valid-frame rate > trajectory_pass_rate."""
        parent = self.parent
        ct = self.conf_thresh
        alive = []
        for t in self.tracks:
            total = len(t.frames)
            if total == 0:
                continue
            valid_count = 0
            for frame in t.frames:
                people = frame.get("people", [])
                person = people[0] if people else parent._build_zero_person_openpose()
                if parent._video_frame_passes_required_points(person, ct):
                    valid_count += 1
            rate = valid_count / float(total)
            if rate > self.trajectory_pass_rate:
                alive.append(t)
        # Sort by validity rate descending (best first)
        def _vr(t):
            n = len(t.frames)
            if n == 0:
                return 0.0
            return sum(1 for f in t.frames
                       if parent._video_frame_passes_required_points(
                           f["people"][0] if f.get("people") else parent._build_zero_person_openpose(), ct)) / float(n)
        alive.sort(key=_vr, reverse=True)
        return alive

    def _find_t_star(self, alive: List[TrackState]) -> int:
        if not alive:
            return -1
        n_tracks = len(alive)
        for fi in range(self.n_frames):
            ok = True
            for t in alive:
                if fi >= len(t.assignment) or t.assignment[fi] < 0:
                    ok = False
                    break
            if ok:
                return fi
        return -1

    def _renumber_at_t_star(self, alive: List[TrackState], t_star: int) -> None:
        """Reorder alive tracks by X coordinate of their reference point at t*."""
        parent = self.parent

        def _sort_key(track: TrackState) -> float:
            # Get the person at t* from the track's frames
            if t_star < len(track.frames):
                people = track.frames[t_star].get("people", [])
                if people:
                    return parent._person_sort_x(people[0], self.conf_thresh)
            return float('inf')

        alive.sort(key=_sort_key)

    def _ensure_first_frame_valid(self, alive: List[TrackState]) -> None:
        """If frame-0 is all-zeros for a track, replace with the first valid frame."""
        parent = self.parent
        zero = parent._build_zero_person_openpose()
        for t in alive:
            if len(t.frames) == 0:
                continue
            f0_people = t.frames[0].get("people", [])
            f0_person = f0_people[0] if f0_people else zero
            if not parent._is_frame_absent(f0_person, self.conf_thresh):
                continue  # frame-0 is fine
            # Find first valid frame
            for i in range(1, len(t.frames)):
                fi_people = t.frames[i].get("people", [])
                fi_person = fi_people[0] if fi_people else zero
                if not parent._is_frame_absent(fi_person, self.conf_thresh):
                    canvas_w = t.frames[0].get("canvas_width", 512)
                    canvas_h = t.frames[0].get("canvas_height", 768)
                    t.frames[0] = build_track_frame(
                        parent._clone_person_fast(fi_person), canvas_w, canvas_h)
                    break

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_candidates(self, frame: dict) -> List[dict]:
        return self.parent._sorted_people_for_frame(frame, self.conf_thresh)

    def _ref_point(self, person: dict, ref_state: dict) -> Optional[np.ndarray]:
        return self.parent._person_reference_point_with_mode(
            self.parent._normalize_person_schema(person),
            self.conf_thresh,
            ref_state,
        )

    def _match_threshold(self, canvas_w: int, canvas_h: int, gap: int = 1) -> float:
        base = self.parent._pixel_match_threshold(canvas_w, canvas_h, ratio=self.match_ratio)
        mult = max(1, min(self.gap_mult_max, gap))
        return base * mult
