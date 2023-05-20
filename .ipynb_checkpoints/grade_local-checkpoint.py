#!/usr/bin/env python3

"""
Do a local practice grading.
The score you recieve here is not an actual score,
but gives you an idea on how prepared you are to submit to the autograder.
"""

import os
import sys

import numpy
import pandas

import cse40.question
import cse40.assignment
import cse40.style
import cse40.utils

class T1A(cse40.question.Question):
    def score_question(self, submission):
        data = pandas.DataFrame({
            'a': [1, 2, 3],
        })

        result = submission.scale_data(data, ['a'])
        if (self.check_not_implemented(result)):
            return

        if (not isinstance(result, pandas.DataFrame)):
            self.fail("Answer must be a DataFrame.")
            return

        self.full_credit()

class T1B(cse40.question.Question):
    def score_question(self, submission):
        data = pandas.DataFrame({
            'a': [1, 2, 3],
        })

        result = submission.split_dict_data(data, 0.5)
        if (self.check_not_implemented(result)):
            return

        if (len(result) != 2):
            self.fail("Must return two objects.")
            return

        for split in result:
            if (self.check_not_implemented(split)):
                return

            if (not isinstance(split, dict)):
                self.fail("Answer must be a dict.")
                return

        self.full_credit()

class T3A(cse40.question.Question):
    def score_question(self, submission):
        result = submission.manual_decision_tree([0.0, 0.0])
        if (self.check_not_implemented(result)):
            return

        if (not isinstance(result, int)):
            self.fail("Answer must be an int.")
            return

        self.full_credit()

class T3B(cse40.question.Question):
    def score_question(self, submission):
        classifier = submission.MyKNN(3)

        features = pandas.DataFrame({
            'a': [1.0, 2.0, 3.0],
        })

        labels = pandas.Series(['x', 'y', 'z'])

        classifier.fit(features, labels)
        result = classifier.predict(features)

        if (self.check_not_implemented(result)):
            return

        if (not isinstance(result, numpy.ndarray)):
            self.fail("Prediction must be a numpy.ndarray.")
            return

        if (result.shape != (3,)):
            self.fail("Prediction does not have the right shape.")
            return

        self.full_credit()

def grade(path):
    submission = cse40.utils.prepare_submission(path)

    questions = [
        T1A("Task 1.A (scale_data)", 1),
        T1B("Task 1.B (split_dict_data)", 1),
        T3A("Task 3.A (manual_decision_tree)", 1),
        T3B("Task 3.B (MyKNN)", 1),
        cse40.style.Style(path, max_points = 1),
    ]

    assignment = cse40.assignment.Assignment('Practice Grading for Hands-On 4', questions)
    assignment.grade(submission)

    return assignment

def main(path):
    assignment = grade(path)
    print(assignment.report())

def _load_args(args):
    exe = args.pop(0)
    if (len(args) != 1 or ({'h', 'help'} & {arg.lower().strip().replace('-', '') for arg in args})):
        print("USAGE: python3 %s <submission path (.py or .ipynb)>" % (exe), file = sys.stderr)
        sys.exit(1)

    path = os.path.abspath(args.pop(0))

    return path

if (__name__ == '__main__'):
    main(_load_args(list(sys.argv)))
