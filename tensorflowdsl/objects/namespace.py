from tensorflowdsl.objects.binding import binding

class namespace (binding):
    def __init__ (self, name):
        super().__init__(name)

        self.__tf_name = name
        self.__children = []

    def add_child (self, child):
        if not isinstance(child, binding):
            raise TypeError('Child of namespace must be a binding.')

        self.__children.append(child)

    def build (self):
        result = 'with tf.name_scope(\'%s\'):\n' % (self.__tf_name, )
        for child in self.__children:
            child_result = child.build()

            for line in child_result.split('\n'):
                result += '    %s\n' % (line, ) 

        return result