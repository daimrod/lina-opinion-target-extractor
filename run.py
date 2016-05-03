import shutil
import os
import crf_absa16
import numpy
import time
import sys
from subprocess import PIPE, Popen


def init(argv):
    """Initialize the enviroment.

    Args:
        argv: sys.argv

    Returns:
        PARTS and WORKDIR

    Raises:
        IOError: An error occurred.
    """
    if len(sys.argv) != 3:
        print('Usage: %s PARTS WORKDIR' % sys.argv[0])
        sys.exit(0)

    parts = int(sys.argv[1])
    workdir = os.path.abspath(sys.argv[2]) + '/'

    shutil.rmtree(workdir, ignore_errors=True)
    os.mkdir(workdir)
    tee = Popen(['tee', workdir + 'log.txt'], stdin=PIPE)
    os.dup2(tee.stdin.fileno(), sys.stdout.fileno())
    os.dup2(tee.stdin.fileno(), sys.stderr.fileno())
    return (parts, workdir)


def full_pipeline_with_fold(parts, workdir,
                            builds, features, shuffle,
                            train_path, test_path, gold_path,
                            tokenizer):
    """Full pipeline.

    Args:
        parts: variable documentation.
        workdir: variable documentation.
        builds: variable documentation.
        features: variable documentation.
        train_path: variable documentation.

    Returns:
        Returns information

    Raises:
        IOError: An error occurred.
    """
    print('')                       # Clean output
    print('Working in ' + workdir)
    n_train = parts - 1
    n_test = 1
    print(time.strftime("%Y-%m-%d %H:%M:%S ") + 'Spliting the dataset...')
    original_train_path = train_path
    full_ident = 'full'
    train_path = os.path.join(workdir, full_ident + '.xml')
    shutil.copyfile(original_train_path, train_path)

    train_indexes, test_indexes = crf_absa16.split_dataset(train_path, parts=parts,
                                                           n_train=n_train, n_test=n_test,
                                                           odir=workdir, shuffle=shuffle)
    print(time.strftime("%Y-%m-%d %H:%M:%S ") + 'Generating features...')
    f_names = crf_absa16.generate_features(features, full_ident, workdir, tokenizer=tokenizer)
    print(time.strftime("%Y-%m-%d %H:%M:%S ") + 'Generating templates...')
    crf_absa16.generate_templates(builds, f_names, workdir=workdir)

    scores = {}
    N = parts
    for n in range(N):
        i = str(n) + '/'
        print(time.strftime("%Y-%m-%d %H:%M:%S ") + 'i = ' + i + str(N))
        print(time.strftime("%Y-%m-%d %H:%M:%S ") + 'Training and evaluating...')
        print(time.strftime("%Y-%m-%d %H:%M:%S ") + 'Extracting sentences for train model...')
        sys.stdout.flush()
        crf_absa16.extract_sentences(workdir + full_ident + '_model.txt',
                                     workdir + i + 'train_model.txt',
                                     train_indexes[n])
        print(time.strftime("%Y-%m-%d %H:%M:%S ") + 'Extracting sentences for test model...')
        sys.stdout.flush()
        crf_absa16.extract_sentences(workdir + full_ident + '_model.txt',
                                     workdir + i + 'test_model.txt',
                                     test_indexes[n])
        crf_absa16.extract_sentences(workdir + full_ident + '_w.txt',
                                     workdir + i + 'test_w.txt',
                                     test_indexes[n])
        crf_absa16.extract_sentences(workdir + full_ident + '_s.txt',
                                     workdir + i + 'test_s.txt',
                                     test_indexes[n])
        for (ident, template) in builds:
            train_file = workdir + i + 'train_model.txt'
            template_file = os.path.join(workdir, ident + '_template.txt')
            test_file = workdir + i + 'test_model.txt'
            gold_xml = workdir + i + 'gold.xml'
            test_xml = workdir + i + 'test.xml'

            if ident not in scores:
                scores[ident] = []
            print(time.strftime("%Y-%m-%d %H:%M:%S ") + 'Pipeline ' + ident + '...')
            sys.stdout.flush()
            scores[ident].append(crf_absa16.pipeline(ident, template_file,
                                                     train_file, test_file,
                                                     gold_xml, test_xml,
                                                     workdir=workdir + i))
        print(time.strftime("%Y-%m-%d %H:%M:%S ") + 'Remove all non XML files...')
        sys.stdout.flush()
        for f in os.listdir(workdir + i):
            if f.endswith('.txt'):
                os.remove(os.path.join(workdir + i, f))

    if test_path is not None and gold_path is not None:
        original_test_path = test_path
        test_path = os.path.join(workdir, 'test.xml')
        shutil.copyfile(original_test_path, test_path)
        crf_absa16.generate_features(features, 'test', workdir,
                                     tokenizer=tokenizer)

        original_gold_path = gold_path
        gold_path = os.path.join(workdir, 'gold.xml')
        shutil.copyfile(original_gold_path, gold_path)
        for (ident, template) in builds:
            print(time.strftime("%Y-%m-%d %H:%M:%S ") + 'Training & Evaluating ' + ident + ' on full')
            sys.stdout.flush()
            template_file = os.path.join(workdir, ident + '_template.txt')
            train_file = workdir + full_ident + '_model.txt'
            test_file = workdir + 'test_model.txt'
            gold_xml = gold_path
            test_xml = test_path
            crf_absa16.pipeline(ident, template_file,
                                train_file, test_file,
                                gold_xml, test_xml,
                                workdir=workdir)

    f1_scores = {}
    pre_scores = {}
    rec_scores = {}
    for ident in scores:
        pre, rec, f1 = [], [], []
        for d in scores[ident]:
            pre.append(d['PRE'])
            rec.append(d['REC'])
            f1.append(d['F-MEASURE'])
            f1_scores[ident] = f1
            pre_scores[ident] = pre
            rec_scores[ident] = rec
    for (s, ident) in sorted([(numpy.mean(f1), ident) for (ident, f1) in f1_scores.items()]):
        print('%s_f1s = %s' % (ident, f1_scores[ident]))
        print('%s_f1 = %f' % (ident, numpy.mean(f1_scores[ident])))
        print('%s_pre = %f' % (ident, numpy.mean(pre_scores[ident])))
        print('%s_rec = %f' % (ident, numpy.mean(rec_scores[ident])))
        sys.stdout.flush()
