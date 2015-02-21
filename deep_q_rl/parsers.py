import argparse

class OtherScriptHelper(argparse.ArgumentParser):
    """
    Thin wrapper that lets you add other argument parsers for which we'll show parameter help as well
    """

    def __init__(self, *args, **kwargs):
        super(OtherScriptHelper, self).__init__(*args, **kwargs)
        self.other_parsers = []


    def format_help(self):
        self_help = super(OtherScriptHelper, self).format_help()
        other_helpers = [parser.format_help() for parser in self.other_parsers]
        return '\n\n'.join([self_help] + other_helpers)
