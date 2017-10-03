from tensorflowdsl.objects.binding import binding
from tensorflowdsl.objects.shared import _split_off_next_symbol

class trainer (binding):
    def __init__ (self, op_syntax):
        self.__tf_name = None
        self.__summary_args = []
        self.__save = None
        self.__learning_rate = None

        op_syntax_head, op_syntax_tail = _split_off_next_symbol(op_syntax)
        if op_syntax_head != 'TRAIN' and op_syntax_head != 'train':
            raise SyntaxError('This should never happen!!')

        op_syntax_head, op_syntax_tail = _split_off_next_symbol(op_syntax_tail)
        self.__type = op_syntax_head
        super().__init__('training_op')

        op_syntax_head, op_syntax_tail = _split_off_next_symbol(op_syntax_tail)
        self.__learning_rate = op_syntax_head


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
        args = ', name=%s' % (self.__tf_name, ) if self.__tf_name is not None else ''

        result = 'training_op = %s(%s%s).minimize(loss)' % (self.__type, self.__learning_rate, args, )
        return result

