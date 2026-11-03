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
