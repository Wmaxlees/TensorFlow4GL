import json
import re
import tensorflow as tf

from tensorflowdsl.objects.input import input_op
from tensorflowdsl.objects.namespace import namespace
from tensorflowdsl.objects.op import op
from tensorflowdsl.objects.trainer import trainer
from tensorflowdsl.objects.var import var



class Runner:
    def __init__ (self, name='AIRunner', output_summary=False):
        self.__name = name
        self.__output_summary = output_summary

        self.__hyperparameters = {}

        self.__bindings = {}

        self.__testing_output = {}
        self.__training_output = {}
        self.__prediction = None
        self.__true_labels = None

        self.__loss = None


    def load_from_file (self, filename):
        with open(filename, 'r') as file:
            model = self.parse_model_file(filename)
            model = model.build()

            print(model)
            # exec('print(tf.contrib.layers)')
            # input()
            exec(model, globals(), locals())


    def load_hyperparameters_from_json (self, filename):
        with open(filename) as hp_file:
            hyperparameters = hp_file.read()
            self.__hyperparameters = json.loads(hyperparameters)


    def set_hyperparameter (self, key, value):
        self.__hyperparameters[key] = value


    def train (self, data, labels, batch_size):
        if self.__sess is None:
            self.__sess = tf.Session()

            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())


    def test (self, data):
        pass


    def parse_model_file (self, filename):
        syntax = None
        with open(filename) as model_file:
            syntax = model_file.read()

        syntax = self.input_hyperparameters(syntax)
        result = self.parse_scope('global', syntax)

        return result

    def input_hyperparameters(self, syntax):
        hyperparameter_locations = [(m.start(0), m.end(0)) for m in re.finditer(r'\$\{[a-zA-Z0-9_]*\}', syntax)]

        offset = 0

        for location in hyperparameter_locations:
            name = syntax[location[0]+2+offset:location[1]-1+offset]
            value = self.__hyperparameters[name]
            
            difference = len(name)+3 - len(str(value))
            
            syntax = syntax[:location[0]+offset] + str(value) + syntax[location[1]+offset:]

            offset -= difference

        return syntax

    def __add_binding (self, scope, new_binding, details):
        for decorator in details['decorators']:
            new_binding.apply_decorator(decorator)

        scope.add_child(new_binding)
        self.__bindings[new_binding.get_name()] = new_binding

        if details['predictor']:
            self.__prediction = new_binding.get_name()
        if details['true_labels']:
            self.__true_labels = new_binding.get_name()
        if details['output'] is not None:
            if details['output'][1] == 'test' or details['output'][1] == 'TEST':
                self.__testing_output[details['output'][0]] = new_binding.get_name()
            elif details['output'][1] == 'train' or details['output'][1] == 'TRAIN':
                self.__training_output[details['output'][0]] = new_binding.get_name()

    def parse_scope (self, scope_name, syntax):
        result = namespace(scope_name)

        lines = syntax.split('\n')

        details = {
            'decorators': [],
            'predictor': False,
            'output': None,
            'true_labels': False
        }

        line_number = 0
        while line_number < len(lines):
            line = lines[line_number].strip()
            if (line.startswith('#')):
                pass

            elif (line.startswith('@')):
                if line == '@PREDICTOR' or line == '@predictor':
                    details['predictor'] = True
                elif line == '@TRUE_LABELS' or line == '@true_labels':
                    details['true_labels'] = True
                elif line.startswith('@OUTPUT') or line.startswith('@output'):
                    _, output_idx, output_type = line.split()

                    details['output'] = (output_idx, output_type, )
                else:
                    details['decorators'].append(line)

            elif (line.startswith('op') or line.startswith('OP')):
                new_op = op(line)

                self.__add_binding(result, new_op, details)

                details = {
                    'decorators': [],
                    'predictor': False,
                    'output': None,
                    'true_labels': False
                }
                

            elif (line.startswith('var') or line.startswith('VAR')):
                new_var = var(line)

                self.__add_binding(result, new_var, details)
                
                details = {
                    'decorators': [],
                    'predictor': False,
                    'output': None,
                    'true_labels': False
                }

            elif (line.startswith('input') or line.startswith('INPUT')):
                new_input = input_op(line)

                self.__add_binding(result, new_input, details)
                
                details = {
                    'decorators': [],
                    'predictor': False,
                    'output': None,
                    'true_labels': False
                }

            elif (line.startswith('use') or line.startswith('USE')):
                tokens = line.split()[1:]
                if tokens[1] != 'IN' and tokens[1] != 'in':
                    raise SyntaxError('USE IN missing IN statement.\n%s' % (line, ))

                self.__bindings[tokens[2]].apply_input(tokens[0])

            elif (line.startswith('pass') or line.startswith('PASS')):
                tokens = line.split()[1:]
                if tokens[1] != 'TO' and tokens[1] != 'to':
                    raise SyntaxError('PASS TO missing TO statement.\n%s' % (line, ))

                self.__bindings[tokens[2]].apply_input(tokens[0])

            elif line.startswith('loss') or line.startswith('LOSS'):
                line = ('OP %s AS loss' % (line[5:], ))
                new_op = op(line)
                new_op.apply_input(self.__prediction)
                new_op.apply_input(self.__true_labels)

                self.__add_binding (result, new_op, details)

                details = {
                    'decorators': [],
                    'predictor': False,
                    'output': None,
                    'true_labels': False
                }

            elif line.startswith('train') or line.startswith('TRAIN'):
                new_trainer = trainer(line)

                self.__add_binding(result, new_trainer, details)

                details = {
                    'decorators': [],
                    'predictor': False,
                    'output': None,
                    'true_labels': False
                }

            elif re.match(r'\'[a-zA-Z_0-9]*\'\:[ \t]+\{', line):
                new_scope = ''
                new_scope_name = line.split('\'')[1]
                while True:
                    line_number += 1
                    line = lines[line_number].strip()
                    if line == '}':
                        result.add_child(self.parse_scope(new_scope_name, new_scope))
                        line_number += 1
                        break
                    else:
                        new_scope = '%s\n%s' % (new_scope, line)


            elif line != '':
                print('UNCAUGHT: %s' % (line, ))

            line_number += 1

        
        return result


if __name__ == '__main__':
    raise Warning('module.py is a library and should not be executed.')
