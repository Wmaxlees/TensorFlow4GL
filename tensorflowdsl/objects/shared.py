from tensorflowdsl.objects.binding import binding

def _split_off_next_symbol (syntax):
    symbols = syntax.split(maxsplit=1)

    if len(symbols) == 2:
        return symbols[0], symbols[1]
    else:
        return symbols[0], ''
