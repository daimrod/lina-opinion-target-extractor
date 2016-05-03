#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re
import sys
import tempfile
import shutil


def paste(files, opath=None, delim='\t'):
    """Merge lines of files.

    Write lines consisting of the sequentially corresponding lines
    from each files, separated by TABs to opath.

    Args:
        files: List of files.
        opath: Output stream.
        delim: Delimiter to join the files.

    Returns:
        Nothing.

    Raises:
        IOError: An error occurred.
    """
    # Create output path
    open(opath, 'w').close()
    p1 = files.pop(0)
    for ipath in files:
        try:
            temp_ofile = tempfile.NamedTemporaryFile('wb')
            i1 = open(p1, 'rb')
            i2 = open(ipath, 'rb')
            for (i1n, i2n) in zip(i1, i2):
                temp_ofile.write(i1n.strip())
                temp_ofile.write(delim)
                temp_ofile.write(i2n.strip())
                temp_ofile.write('\n')
        except Exception as ex:
            print(ex)
        finally:
            temp_ofile.flush()
            shutil.copyfile(temp_ofile.name, opath)
            p1 = opath
            temp_ofile.close()
            i1.close()
            i2.close()


def spans_retrieval(raw_text, tokenized_text, separator=' '):
    """Find the spans in raw_text of tokens in tokenized_text.

    The spans are the indices of the tokens found in the tokenized
    text. We assume that the tokens are separated by a separator (a
    whitespace by default) in tokenized_text.

    The purpose of this tool is to retrieve the spans of tokens from
    the tokenized text and the original (raw) text. This is useful
    when you need to annotate text and produce meaningful results like
    an XML... (e.g. SEM16).

    Args:
        raw_text: The text original that has been tokenized.
        tokenized_text: The tokenized text.
        separator: The string used to separate tokens in tokenized_text.

    Returns:
        A list of spans, a span is a tuple of integer (start, end).
        The token can be retrived from the raw_text with
        raw_text[start:end].

    Raises:
        IOError: An error occurred.
    """
    ret = []
    pos = 0
    for token in re.split(separator, tokenized_text):
        m = re.search(re.escape(token), raw_text[pos:])
        ret.append((m.start() + pos, m.end() + pos))
        pos = m.end()

    return ret


def cut(ipath, fields, opath=None):
    """Remove the sections from each lines of file.

    Print selected parts of lines from file. Fields start at 0.

    Args:
        ipath: Input file.
        fields: List of fields to select.
        opath: Output file.

    Returns:
        Nothing.

    Raises:
        IOError: An error occurred.
    """
    try:
        ifile = open(ipath, 'rb')
        ofile = None
        if opath is None:
            ofile = sys.stdout
        else:
            ofile = open(opath, 'wb')
        for line in ifile:
            line = line.strip()
            tab = line.split()
            if tab != []:
                ofile.write('\t'.join([tab[f] for f in fields]))
            ofile.write('\n')
    finally:
        ifile.close()
        if ofile != sys.stdout and ofile is not None:
            ofile.close()
