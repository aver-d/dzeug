# dzeug

dzeug provides command line tools I use for working with German language texts.

Texts can be parsed, stored and used with a small search engine for finding sentences and vocabulary based on lemmatized word forms.

Api is unstable. Not currently for general use.

    Usage: dzeug-sent [OPTIONS] WORD [PATHS]...

    Options:
      -c, --centre     Align word centrally when displaying
      -s, --sentences  Show only sentences, no document titles
      -t, --stdout     No pager, send directly to terminal
      -z, --shuffle    Shuffle results order
      --no-retry       Exit immediately on not found
      --help           Show this message and exit.


    Usage: dzeug-vocab [OPTIONS] [PATH]

    Options:
      -n, --nouns       Include nouns
      -v, --verbs       Include verbs
      -a, --adjectives  Include adjectives
      -r, --range TEXT  Limit vocab rank to this range [1,45000]
      --simple          Show only lemmas as simple list, no formatting
      -d, --dzeug       Input text is already tagged in dzeug format
      --help            Show this message and exit.


    Usage: dzeug-pos [OPTIONS] [FILES]...

    Options:
    -s, --save       For each named file provided, save its corresponding parsed
                     document. The file is saved in the directory given by the
                     config file's 'dir.parsed' value, and the filename is saved
                     with an additional .dzeug extension.

                     Default: output is sent to stdout and input files are separated
                     with two empty lines.

    --no-lang-check  Skip language detection
    --help           Show this message and exit.
