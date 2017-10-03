from tensorflowdsl.objects.binding import binding
from tensorflowdsl.objects.shared import _split_off_next_symbol

class op (binding):
    def __init__ (self, op_syntax):
        self.__tf_name = None
        self.__summary_args = []
        self.__save = None
        self.__args = ()
        self.__inputs = {}
        self.__activation = None

        op_syntax_head, op_syntax_tail = _split_off_next_symbol(op_syntax)
        if op_syntax_head != 'OP' and op_syntax_head != 'op':
            raise SyntaxError('This should never happen!!')

        op_syntax_head, op_syntax_tail = _split_off_next_symbol(op_syntax_tail)
        self.__type = op_syntax_head

        op_syntax_head, op_syntax_tail = _split_off_next_symbol(op_syntax_tail)
        if op_syntax_head != 'AS' and op_syntax_head != 'as':
            raise SyntaxError('OP syntax invalid. Requires AS statement.\n%s' % (op_syntax, ))

        op_syntax_head, op_syntax_tail = _split_off_next_symbol(op_syntax_tail)
        super().__init__(op_syntax_head)

        # op_syntax_head, op_syntax_tail = op_syntax_tail.split(maxsplit=1)
        if op_syntax_tail.startswith('ARGS') or op_syntax_tail.startswith('args'):
            # Split off ARGS and parenthesis
            args = op_syntax_tail[5:-1]
            args = args.split(',')
            for arg in args:
                self.__args = self.__args + (arg.strip(), )

        elif op_syntax_tail != '':
            raise SyntaxError('Invalid ARGS statement.\n%s' % (op_syntax, ))


    def apply_decorator (self, decorator_syntax):
        # Remove the leading @
        decorator_syntax = decorator_syntax[1:]

        # Tokenize
        tokens = decorator_syntax.split()

        # For convenience
        decorator = tokens[0].lower()

        # Apply decorator details
        if decorator == 'save':
            self.__save = tokens[1]
        elif decorator == 'name':
            self.__tf_name = tokens[1]
        elif decorator == 'summarize':
            self.__summary_args.append((tokens[1], tokens[2], ))
        elif decorator == 'relu':
            self.__activation = 'tf.nn.relu'
        else:
            raise SyntaxError('Unknown decoration for %s: %s' % (super().get_name(), tokens[0], ))

    def apply_input (self, input_name, input_idx=None):
        if input_idx is None:
            # THIS IS DEFINITELY GOING TO BREAK IF PEOPLE USED MIXED
            input_idx = len(self.__inputs)

        self.__inputs[input_idx] = input_name
    
    def build (self):
        # Build arguments
        arg_list = ''
        for idx in range(len(self.__inputs)):
            arg_list = '%s%s, '  % (arg_list, self.__inputs[idx], )

        for arg in self.__args:
            arg_list = '%s%s, ' % (arg_list, arg, )


        if self.__tf_name is not None:
            arg_list = '%sname=%s, ' % (arg_list, self.__tf_name, )

        result = '%s(%s)' % (self.__type, arg_list[:-2], )
        if self.__activation is not None:
            result = '%s(%s)' % (self.__activation, result)
        result = '%s = %s' % (super().get_name(), result, )
        return result

