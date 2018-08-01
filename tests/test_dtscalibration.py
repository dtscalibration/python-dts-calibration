# coding=utf-8
from dtscalibration.cli import main


def test_main():
    assert main([]) == 0
