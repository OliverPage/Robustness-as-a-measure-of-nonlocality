#!/bin/bash
python OptRobustnessVsN.py 2 2 10000 'calculate'
python OptRobustnessVsN.py 2 3 10000 'calculate'
python OptRobustnessVsN.py 2 4 10000 'calculate'
python OptRobustnessVsN.py 2 5 10000 'calculate'
python plotOptRvsN.py 2 2 10000
python OptRobustnessVsN.py 3 2 10000 'calculate'
python OptRobustnessVsN.py 3 3 10000 'calculate'
python OptRobustnessVsN.py 3 4 10000 'calculate'
python OptRobustnessVsN.py 3 5 10000 'calculate'
python plotOptRvsN.py 3 2 10000