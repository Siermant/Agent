import unittest
from global_agent.presentation import request_path, equity_chart

class PresentationTests(unittest.TestCase):
    def test_query_routes(self):
        self.assertEqual(request_path('/?utm_source=chatgpt.com'), '/')
        self.assertEqual(request_path('/api/status?x=1'), '/api/status')
        self.assertEqual(request_path('/api/execute?x=1'), '/api/execute')
        self.assertEqual(request_path('https://example.com/'), '')
        self.assertEqual(request_path('//example.com/'), '')
    def test_empty(self):
        self.assertIn('Brak zapisanych', equity_chart([]))
    def test_gap_and_invalid(self):
        row={'account':'A:crypto','initial_pln':'20000','history':[
            {'day':'2026-11-25','equity':'20000'},
            {'day':'2026-11-27','equity':'20200'},
            {'day':'2026-11-28','equity':'NaN'},
            {'day':'<script>','equity':'20000'}]}
        chart=equity_chart([row])
        self.assertEqual(chart.count('<circle'),2)
        self.assertNotIn('stroke-width="2"',chart)
        self.assertNotIn('<script>',chart)
        self.assertIn('+1.000%',chart)
    def test_consecutive(self):
        row={'account':'B:crypto','initial_pln':'100','history':[
            {'day':'2026-11-26','equity':'99'}, {'day':'2026-11-27','equity':'101'}]}
        self.assertIn('stroke-width="2"',equity_chart([row]))
