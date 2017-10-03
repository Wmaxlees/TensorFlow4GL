import json
import re

from tensorflowdsl.objects.input import input_op
from tensorflowdsl.objects.namespace import namespace
from tensorflowdsl.objects.op import op
from tensorflowdsl.objects.var import var



class Runner:
    def __init__ (self, name='AIRunner', output_summary=False):
        self.__name = name
        self.__output_summary = output_summary

        self.__hyperparameters = {}

        self.__bindings = {}

        self.__output = {}
        self.__prediction = None
        self.__true_labels = None


    def load_from_file (self, filename):
        with open(filename, 'r') as file:
            model = self.parse_model_file(filename)
            model = model.build()

            print(model)
            exec(model)


    def load_hyperparameters_from_json (self, filename):
        with open(filename) as hp_file:
            hyperparameters = hp_file.read()
            self.__hyperparameters = json.loads(hyperparameters)


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

    def parse_scope (self, scope_name, syntax):
        result = namespace(scope_name)

        lines = syntax.split('\n')

        current_decorators = []
        predictor = False
        output = None
        true_labels = False
        line_number = 0
        while line_number < len(lines):
            line = lines[line_number].strip()
            if (line.startswith('#')):
                pass

            elif (line.startswith('@')):
                if line == '@PREDICTOR' or line == '@predictor':
                    predictor = True
                elif line == '@TRUE_LABELS' or line == '@true_labels':
                    true_labels = True
                elif line.startswith('@OUTPUT') or line.startswith('@output'):
                    output_idx = line.split()[1]
                    output = output_idx
                else:
                    current_decorators.append(line)

            elif (line.startswith('op') or line.startswith('OP')):
                new_op = op(line)

                for decorator in current_decorators:
                    new_op.apply_decorator(decorator)
                current_decorators = []

                result.add_child(new_op)
                self.__bindings[new_op.get_name()] = new_op

                if predictor:
                    self.__prediction = new_op.get_name()
                    predictor = False
                if true_labels:
                    self.__true_labels = new_op.get_name()
                    true_labels = False
                if output is not None:
                    self.__output[output] = new_op.get_name()
                    output = None

            elif (line.startswith('var') or line.startswith('VAR')):
                new_var = var(line)

                for decorator in current_decorators:
                    new_var.apply_decorator(decorator)
                current_decorators = []

                result.add_child(new_var)
                self.__bindings[new_var.get_name()] = new_var

                if predictor:
                    self.__prediction = new_var.get_name()
                    predictor = False
                if true_labels:
                    self.__true_labels = new_var.get_name()
                    true_labels = False
                if output is not None:
                    self.__output[output] = new_var.get_name()
                    output = None

            elif (line.startswith('input') or line.startswith('INPUT')):
                new_input = input_op(line)

                for decorator in current_decorators:
                    new_input.apply_decorator(decorator)
                current_decorators = []

                result.add_child(new_input)
                self.__bindings[new_input.get_name()] = new_input

                if predictor:
                    self.__prediction = new_input.get_name()
                    predictor = False
                if true_labels:
                    self.__true_labels = new_input.get_name()
                    true_labels = False
                if output is not None:
                    self.__output[output] = new_input.get_name()
                    output = None

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

            elif re.match(r'\'[a-zA-Z_]*\'\:[ \t]+\{', line):
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
