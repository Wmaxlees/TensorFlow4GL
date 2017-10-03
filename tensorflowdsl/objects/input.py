from tensorflowdsl.objects.binding import binding

def _split_off_next_symbol (syntax):
    symbols = syntax.split(maxsplit=1)

    if len(symbols) == 2:
        return symbols[0], symbols[1]
    else:
        return symbols[0], ''

class input_op (binding):
    def __init__ (self, op_syntax):
        self.__tf_name = None
        self.__summary_args = []
        self.__save = None
        self.__shape = ()

        op_syntax_head, op_syntax_tail = _split_off_next_symbol(op_syntax)
        if op_syntax_head != 'INPUT' and op_syntax_head != 'input':
            raise SyntaxError('This should never happen!!')

        op_syntax_head, op_syntax_tail = _split_off_next_symbol(op_syntax_tail)
        super().__init__(op_syntax_head)

        if op_syntax_tail.startswith('SIZE') or op_syntax_tail.startswith('size'):
            # Split off ARGS and parenthesis
            size = op_syntax_tail[5:-1]
            size = size.split(',')
            for dim in size:
                if dim != '' and dim != ' ':
                    self.__shape = self.__shape + (dim.strip(), )

        else:
            raise SyntaxError('VAR syntax invalid. Requires SIZE statement.\n%s' % (op_syntax, ))


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
        else:
            raise SyntaxError('Unknown decoration for %s: %s' % (super().get_name(), tokens[0], ))

    
    def build (self):
        # Build arguments
        arg_list = 'tf.float32, (' # SHOULD BE A DECORATOR TO CHANGE THIS

        for dim in self.__shape:
            arg_list = '%s%s, ' % (arg_list, dim, )
        arg_list = '%s), ' % (arg_list, )


        if self.__tf_name is not None:
            arg_list = '%sname=%s, ' % (arg_list, self.__tf_name, )

        result = '%s = tf.placeholder(%s)' % (super().get_name(), arg_list[:-2])
        return result

