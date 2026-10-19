import unittest
from src.nlp_parser import parse_bike_request

class NLPParserTests(unittest.TestCase):
    def test_example_request(self):
        result = parse_bike_request("Predict bike demand on a summer weekday at 8 AM, 25°C, with 60% humidity and clear weather.")
        fields = result["fields"]
        self.assertEqual(fields["season_label"], "Summer")
        self.assertEqual(fields["hour"], 8)
        self.assertEqual(fields["temperature"], 25.0)
        self.assertEqual(fields["humidity"], 60.0)
        self.assertEqual(fields["weather_label"], "Clear / partly cloudy")
        self.assertEqual(fields["workingday_label"], "Yes")
        self.assertEqual(fields["month"], 7)

    def test_pm_and_wind(self):
        result = parse_bike_request("winter on Friday at 6 pm, 12 C, humidity 40%, wind speed 15 km/h, light rain")
        fields = result["fields"]
        self.assertEqual(fields["hour"], 18)
        self.assertEqual(fields["weekday_label"], "Friday")
        self.assertEqual(fields["wind"], 15.0)
        self.assertEqual(fields["weather_label"], "Light rain / snow")

if __name__ == "__main__":
    unittest.main()
