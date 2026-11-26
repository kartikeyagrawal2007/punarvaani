"""Dependency-free tests for mouth ROI box maths."""

import unittest

from mouth_geometry import center_square_box, compute_mouth_box, landmark_pixels


class TestMouthGeometry(unittest.TestCase):
    def test_box_contains_points_and_is_square(self):
        points = [(100, 100), (140, 100), (100, 130), (140, 130)]  # 40 x 30
        top, bottom, left, right = compute_mouth_box(points, 320, 240, square_scale=1.3)
        self.assertLessEqual(left, 100)
        self.assertGreaterEqual(right, 140)
        self.assertLessEqual(top, 100)
        self.assertGreaterEqual(bottom, 130)
        self.assertEqual(right - left, bottom - top)

    def test_box_is_clamped_to_image(self):
        points = [(-20, -10), (10, 10)]
        top, bottom, left, right = compute_mouth_box(points, 100, 80)
        self.assertGreaterEqual(top, 0)
        self.assertGreaterEqual(left, 0)
        self.assertLessEqual(bottom, 79)
        self.assertLessEqual(right, 99)

    def test_ema_smooths_towards_previous_box(self):
        points = [(100, 100), (120, 120)]
        first = compute_mouth_box(points, 300, 300)
        second = compute_mouth_box(points, 300, 300, prev_box=first, alpha=0.25)
        self.assertEqual(second, first)

        moved = [(160, 160), (180, 180)]
        smoothed = compute_mouth_box(moved, 300, 300, prev_box=first, alpha=0.25)
        unsmoothed = compute_mouth_box(moved, 300, 300)
        self.assertLess(abs(smoothed[0] - first[0]), abs(unsmoothed[0] - first[0]))
        self.assertGreater(abs(smoothed[0] - first[0]), 0)

    def test_center_square_box(self):
        top, bottom, left, right = center_square_box(200, 100)
        self.assertEqual(right - left, bottom - top)
        self.assertEqual(right - left, 99)

    def test_landmark_pixels_accepts_objects_and_dicts(self):
        class Landmark:
            def __init__(self, x, y):
                self.x, self.y = x, y

        landmarks = [Landmark(0.1, 0.2), {"x": 0.5, "y": 0.6}]
        points = landmark_pixels(landmarks, indices=[0, 1], image_width=200, image_height=100)
        self.assertEqual(points, [(20.0, 20.0), (100.0, 60.0)])

    def test_empty_points_raise(self):
        with self.assertRaises(ValueError):
            compute_mouth_box([], 100, 100)


if __name__ == "__main__":
    unittest.main()


class TestStableCropTrack(unittest.TestCase):
    def test_gaps_are_filled_and_track_is_smoothed(self):
        import numpy as np

        from mouth_geometry import stable_crop_track

        n = 20
        centers = [(100.0 + (5.0 if i % 2 else -5.0), 200.0) for i in range(n)]  # 10 px landmark jitter
        valid = [i not in (3, 4) for i in range(n)]
        cx, cy, roll, side = stable_crop_track(centers, [100.0] * n, [0.0] * n, valid, window=9)
        self.assertLess(np.abs(np.diff(cx)).max(), 2.0)  # the jitter is smoothed away
        self.assertTrue(np.allclose(cy, 200.0) and np.allclose(side, 120.0))  # 1.2 x eye distance

    def test_size_follows_eyes_not_lips(self):
        from mouth_geometry import stable_crop_track

        n = 12
        track = stable_crop_track([(50.0, 50.0)] * n, [80.0] * 6 + [100.0] * 6, [0.0] * n, [True] * n, window=1)
        self.assertAlmostEqual(track[3][0], 96.0)   # camera zoom: eyes 80 -> 100 px, crop 96 -> 120 px
        self.assertAlmostEqual(track[3][-1], 120.0)

    def test_too_many_missing_frames_gives_none(self):
        from mouth_geometry import stable_crop_track

        self.assertIsNone(stable_crop_track([(0, 0)] * 10, [50.0] * 10, [0.0] * 10, [True] * 4 + [False] * 6))

    def test_affine_centres_mouth_scales_and_undoes_roll(self):
        import numpy as np

        from mouth_geometry import crop_affine

        m = crop_affine(300.0, 400.0, 0.0, 112.0, 224)
        self.assertTrue(np.allclose(m @ [300.0, 400.0, 1.0], [112.0, 112.0]))      # mouth -> centre
        self.assertTrue(np.allclose(m @ [356.0, 400.0, 1.0], [224.0, 112.0]))      # half a side -> edge
        tilted = crop_affine(0.0, 0.0, 30.0, 224.0, 224)
        p = np.array([np.cos(np.radians(30)), np.sin(np.radians(30))]) * 50          # a point along the tilted eye line
        out = tilted @ [p[0], p[1], 1.0]
        self.assertAlmostEqual(out[1], 112.0, places=6)                             # lands level with the centre
