"""The main module for nasctn_ai.

This software was developed by employees of the National Institute of Standards
and Technology (NIST), an agency of the Federal Government. Pursuant to title 17
United States Code Section 105, works of NIST employees are not subject to
copyright protection in the United States and are considered to be in the public
domain. Permission to freely use, copy, modify, and distribute this software and its
documentation without fee is hereby granted, provided that this notice and disclaimer
of warranty appears in all copies.

THE SOFTWARE IS PROVIDED 'AS IS' WITHOUT ANY WARRANTY OF ANY KIND, EITHER EXPRESSED,
IMPLIED, OR STATUTORY, INCLUDING, BUT NOT LIMITED TO, ANY WARRANTY THAT THE SOFTWARE
WILL CONFORM TO SPECIFICATIONS, ANY IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE, AND FREEDOM FROM INFRINGEMENT, AND ANY WARRANTY THAT THE
DOCUMENTATION WILL CONFORM TO THE SOFTWARE, OR ANY WARRANTY THAT THE SOFTWARE WILL BE
ERROR FREE. IN NO EVENT SHALL NIST BE LIABLE FOR ANY DAMAGES, INCLUDING, BUT NOT
LIMITED TO, DIRECT, INDIRECT, SPECIAL OR CONSEQUENTIAL DAMAGES, ARISING OUT OF,
RESULTING FROM, OR IN ANY WAY CONNECTED WITH THIS SOFTWARE, WHETHER OR NOT BASED UPON
WARRANTY, CONTRACT, TORT, OR OTHERWISE, WHETHER OR NOT INJURY WAS SUSTAINED BY PERSONS
OR PROPERTY OR OTHERWISE, AND WHETHER OR NOT LOSS WAS SUSTAINED FROM, OR AROSE OUT OF
THE RESULTS OF, OR USE OF, THE SOFTWARE OR SERVICES PROVIDED HEREUNDER.

Distributions of NIST software should also include copyright and licensing statements
of any third-party software that are legally bundled with the code in compliance with
the conditions of those licenses.
"""

import json
import tensorflow as tf


class AIRunner:
    def __init__ (self, name='AIRunner', output_summary=False):
        self.__name = name
        self.__output_summary = output_summary

        self.__hyperparameters = {}

        self.__training_summaries = []
        self.__testing_summaries = []

        self.__sess = None


    def load_from_file (self, filename):
        with open(filename, 'r') as file:


    def load_hyperparameters_from_json (self, filename):
        with open(filename) as hp_file:
            hyperparameters = hp_file.read()
            hyperparameters = json.loads(hyperparameters)


    def set_hyperparameter (self, key, value):
        self.__hyperparameters[key] = value


    def set_optimization_fn (self, fn, learning_rate=0.1):
        pass


    def train (self, data, labels, batch_size):
        if self.__sess is None:
            self.__sess = tf.Session()

            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())




    def test (self, data):
        pass


def __parse_model_file (filename):
    with open(filename) as model_file:
        line = model_file.read_line()

        while line != '' or line is not None:
            print(line)
            line = model_file.read_line()


if __name__ == '__main__':
    raise Warning('module.py is a library and should not be executed.')
