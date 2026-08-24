import unittest
from unittest.mock import MagicMock
from agent.engine.planner import CurriculumPlanner

class TestCurriculumPlanner(unittest.TestCase):
    def test_plan_curriculum_success(self):
        mock_brain = MagicMock()
        mock_brain.generate.return_value = '["Unit 1", "Unit 2", "Unit 3"]'
        mock_brain.extract_json.return_value = ["Unit 1", "Unit 2", "Unit 3"]
        
        planner = CurriculumPlanner(mock_brain)
        units = planner.plan_curriculum("test topic")
        
        self.assertEqual(len(units), 3)
        self.assertEqual(units[0], "Unit 1")

if __name__ == "__main__":
    unittest.main()
