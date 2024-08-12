import argparse

from qnlp import QNLP

argparser = argparse.ArgumentParser()
argparser.add_argument("--testing", action="store_true")

args = argparser.parse_args()

qnlp = QNLP(testing=args.testing)

qnlp.run()
