from __future__ import print_function, division
from psychopy import visual, event, core, logging, gui, data
from psychopy.tools.filetools import fromFile, toFile
import numpy as np
import pandas as pd
import random
import csv
import math
import itertools
from PIL import Image
import PIL.ImageOps
import scipy


stim_list = [3, '3i', 4, '4i', 5, '5i', 6, '6i', 23, 34, 36, 51, 58, 60, 78, 92]
basepairs = [(3, '3i'), (4, '4i'), (5, '5i'), (6, '6i'),
             (23, 34), (36, 51), (58, 60), (78, 92)]


StartUp = gui.Dlg(title="Fractal Experiment")
StartUp.addField('Subject Number:', 'test')
StartUp.addField('Actual Data:', True)
StartUp.addText('')
StartUp.addText('Choose outcome task:')
StartUp.addField('Numerosity?:', False)
StartUp.addField('Associative Inference?:', False)
StartUp.addField('New Associates?:', False)
StartUp.addText('')
StartUp.addText('Outcome task before AND after?:')
StartUp.addField('Numerosity?:', True)
StartUp.addField('Associative Inference?:', False)
StartUp.addField('New Associates?:', False)
StartUp.addText('')
StartUp.addField('# Template Runs?:', 0)
StartUp.addField('# Stat Learning Runs?:', 6)
StartUp.addField('# Outcome Runs?:', 1)
StartUp.addField('# Pair Exposures (Stat/Template):', 5)
StartUp.addField('# Frequency Judgments (Numerosity):', 10)
StartUp.addField('# Repetitions (Associative Tasks):', 1)
StartUp.addField('% Rectangles (Stat/Template):', 10)
StartUp.show()
if StartUp.OK:
    starters = StartUp.data
    subid, check, numer, associnf, newassoc, numer1, associnf1, newassoc1, templates, stats, outcomes, exposures, nums, encodings, rectangles = starters
    check_tasks = np.array([numer, associnf, newassoc])
    task_times = np.array([])
    task_names = np.array(['numerosity', 'associative_inf', 'new_associates'])
    tasks = list(task_names[check_tasks])
    _befaft = np.array([numer1, associnf1, newassoc1])
    befaft = list(_befaft[check_tasks])
    _times = np.array([58, 43, 69, 97, 69, 97])
    multipliers = np.array([exposures, nums, encodings, 1, encodings, 1])
    __total_time = _times * multipliers
    _total_time = np.array([__total_time[0], __total_time[1], np.sum(__total_time[2:4]), np.sum(__total_time[4:6])])
    trim = np.insert(np.copy(check_tasks), 0, True)
    total_time = _total_time * trim
    total_time[0] *= (stats + (templates * 2))
    total_time[1:] *= outcomes
    prepost = np.insert(np.array([2 if i else 1 for i in list(_befaft)]), 0, 1)
    print(total_time)
    print(prepost)
    total_time *= prepost
    time = np.sum(total_time)/60
    print("{} templating runs, {} stat runs, {} outcome runs: {}".format(templates, stats, outcomes, tasks))
    print("Stat learning exposures per run: {}".format(exposures)) if stats > 0 else print('', end='')
    print("Encoding reps per item: {}".format(encodings)) if any(t for t in check_tasks[1:]) else print('', end='')
    print("Numerosity trials per item: {}".format(nums)) if check_tasks[0] else print('', end='')
    print("Estimated time to completion: {} minutes".format(np.around(time, 2)))
    Confirm = gui.Dlg(title="Confirmation")
    Confirm.addText('')
    Confirm.addText("Tasks:", "{}".format(tasks))
    Confirm.addFixedField("Templating runs:", "{}".format(templates))
    Confirm.addFixedField("Stat Learning runs:", "{}".format(stats))
    Confirm.addFixedField("Outcome runs:", "{}".format(outcomes))
    Confirm.addText('')
    Confirm.addFixedField("Stat learning exposures per run:", "{}".format(exposures)) if stats > 0 else Confirm.addText('')
    Confirm.addFixedField("Encoding reps per item:", "{}".format(encodings)) if any(t for t in check_tasks[1:]) else Confirm.addText('')
    Confirm.addFixedField("Numerosity trials per item:", "{}".format(nums)) if check_tasks[0] else Confirm.addText('')
    Confirm.addText('')
    Confirm.addText("Estimated time to completion: {} minutes, plus instructions".format(np.around(time, 2)))
    Confirm.show()
    if Confirm.OK:
        pass
    else:
        core.quit()
else:
    core.quit()



rawTimer = core.Clock()
trialTimer = core.Clock()

mywin = visual.Window([1000, 750], monitor="testMonitor", units="pix", fullscr=check)
mywin.mouseVisible = False if check else True


def quick_exit():
    if 'escape' in event.getKeys():
        core.quit()


def instruction_screen(text, width, delay, last=False, done=False):
    quick_exit()
    mywin.flip()
    instruct = visual.TextStim(mywin, text=text, wrapWidth=width,
                               alignHoriz='left', pos=(-(width/2), 100))
    if done:
        wrap = ''
    else:
        wrap = 'When you are ready to begin, press SPACE' if last else 'Press SPACE to continue'
    color = (51, 204, 51) if last else 'white'
    bold = True if last else False
    space = visual.TextStim(mywin, text=wrap, wrapWidth=width, colorSpace='rgb255',
                            alignHoriz='center', color=color, pos=(0, -100), bold=bold)
    instruct.draw()
    space.draw()
    mywin.update()
    NoKey = True
    while NoKey:
        allKeys = event.getKeys()
        if len(allKeys) > 0:
            resp = allKeys[0]
            if resp == 'space':
                NoKey = False
    mywin.flip()
    core.wait(delay)


# Generates ISIs
def gen_isi(exposures):
    num_trials = exposures * 16
    modifier = int(math.ceil(num_trials / 5))
    ones = [1] * modifier * 2
    threes = [3] * modifier * 2
    fives = [5] * modifier
    ISIs = itertools.chain(ones, threes, fives)
    isi_list = list(ISIs)
    random.shuffle(isi_list)
    return isi_list


def gen_trial_new_assoc(exposures, pairset):
    stims = []
    pairs = [i for i in pairset]
    for i in pairs:
        stims.extend(i)
    all_stim = np.setdiff1d(np.array(range(1, 102)), np.array([i for i in stims if type(i) is int]))
    new_assoc = np.random.choice(all_stim, 24, replace=False)
    new_pairs = [(new_assoc[i - 1], new_assoc[i]) for i in range(1, 17, 2)]
    inf_pairs = list(np.roll(np.copy(new_pairs), 2))
    lures = [(new_assoc[i - 1], new_assoc[i]) for i in range(17, 25, 2)]
    lures2 = list(np.roll(np.copy(lures), 1))
    lures2 = [(i[0], i[1]) for i in lures2]
    lures.extend(lures2)
    _test_list = []
    enc_list = []
    for i in range(len(pairs)):
        _test_list.append((pairs[i][0], new_pairs[i][0], new_pairs[i][1],
                          lures[i][0], inf_pairs[i][0], pairs[i], new_pairs[i], lures[i]))
        _test_list.append((pairs[i][1], new_pairs[i][1], new_pairs[i][0],
                          lures[i][1], inf_pairs[i][1], pairs[i], new_pairs[i], lures[i]))
    test_list = _test_list
    random.shuffle(test_list)
    for j in range(exposures):
        shuffled = _test_list
        random.shuffle(shuffled)
        if j != 0:
            while shuffled[0] == lastitem:
                random.shuffle(shuffled)
        enc_list.extend(shuffled)
        lastitem = shuffled[-1]
    vis_sim = ['sim' if 'i' in str(item[5]) else 'not' for item in test_list]
    return enc_list, test_list, vis_sim


# Generates trial list for numerosity task
def gen_trial_num(trials_per_pair, pairset):
    new_list = []
    pairs = [i for i in pairset]
    probe_sel = [0, 1] * math.ceil(trials_per_pair * len(pairs) / 2)
    hi_freq = ['targ', 'targ', 'fill', 'fill'] * math.ceil(trials_per_pair * len(pairs) / 4)
    for j in pairs:
        j1, j2 = j
        if 'i' not in str(j2):
            temp = [x for x in pairs[:4] if x != j]
        else:
            temp = [x for x in pairs[4:] if x != j]
        all = []
        num_lists = int(math.ceil(trials_per_pair / len(temp)))
        for w in range(num_lists):
            _all = range(0, len(temp))
            random.shuffle(_all)
            all.extend(_all)
        for k in all[0:trials_per_pair]:
            new = np.random.randint(1, len(temp)) + k
            if new > len(temp) - 1:
                new = new - len(temp)
            pmnum = [0, 1]
            random.shuffle(pmnum)
            k1 = temp[all[k]][pmnum[0]]
            if 'i' not in str(j2):
                k2 = temp[new][pmnum[1]]
            else:
                k2 = str(k1) + 'i'
            new_list.append([j1, j2, k1, k2])
    zipper = zip(probe_sel, hi_freq, new_list)
    while any(i[2][0] == j[2][0] for i, j in zip(zipper, zipper[1:])):
        random.shuffle(zipper)
    vis_sim = ['sim' if 'i' in str(item[2][:2]) else 'not' for item in zipper]
    return zipper, vis_sim


# Generates trial sequence for stat learning
def gen_trial_stat(exposures, pairset):
    pairs = [i for i in pairset]
    _trial_list = []
    trial_list = []
    pair_list = []
    for j in range(exposures):
        shuffled = pairs
        random.shuffle(shuffled)
        if j != 0:
            while shuffled[0] == lastitem:
                random.shuffle(shuffled)
        _trial_list.extend(shuffled)
        lastitem = shuffled[-1]
    while any(i == j for i, j in zip(_trial_list, _trial_list[1:])):
        assert False
    for i in _trial_list:
        trial_list.extend(i)
        pair_list.append(i)
        pair_list.append(i)
    return trial_list, pair_list


# Generates trials for perceptual templating
def gen_trial_prepost(exposures, pairset):
    pairs = [i for i in pairset]
    __trial_list = pairs
    _trial_list = []
    trial_list = []
    _pair_list = []
    pair_list = []
    for i in __trial_list:
        _trial_list.extend(i)
        _pair_list.append(i)
        _pair_list.append(i)
    for j in range(exposures):
        shuffled = zip(_trial_list, _pair_list)
        random.shuffle(shuffled)
        if j != 0:
            while shuffled[0] == lastitem:
                random.shuffle(shuffled)
        t_l, p_l = zip(*shuffled)
        trial_list.extend(t_l)
        pair_list.extend(p_l)
        lastitem = shuffled[-1]
    while any(i == j for i, j in zip(trial_list, trial_list[1:])):
        assert False
    return trial_list, pair_list


# Generates trials for the cover task
def gen_cover(exposures, percent):
    num_trials = exposures * 16
    num_patches = int(math.floor(num_trials / 100 * percent))
    num_free = num_trials - num_patches
    rect_list = [False] * num_free
    rect_list2 = [True] * num_patches
    rect_list.extend(rect_list2)
    random.shuffle(rect_list)
    answers = ['left' if i else 'right' for i in rect_list]
    random.shuffle(stim_list)
    return answers, rect_list


# Produces inverted fractal
def invert_image(image):
    im = Image.open('Anna 2012 Fractals/' + str(image) + '.tiff')
    r, g, b, a = im.split()
    rgb_image = Image.merge('RGB', (r, g, b))
    inverted = PIL.ImageOps.invert(rgb_image)
    r2, g2, b2 = inverted.split()
    final = Image.merge('RGBA', (r2, g2, b2, a))
    final.save('Anna 2012 Fractals/' + str(image) + 'i.tiff')


# Takes an image number and generates the path, then draws it
def gen_image(image, position=[0, 0], size=None):
    curr_stim = visual.ImageStim(mywin,
                                 image='Anna all Fractals/' + str(image) + '.tiff',
                                 pos=position, size=size)
    curr_stim.draw()


# Generates a rectangular patch in the bound of the fractal
def gen_patch():
    angle = np.random.uniform(0, 360)
    distance = np.random.uniform(0, 56)
    newx = np.round(math.cos(angle * math.pi / 180) * distance, 0)
    newy = np.round(math.sin(angle * math.pi / 180) * distance, 0)
    patch = visual.Rect(mywin, width=16, height=16, pos=(newx, newy),
                        fillColor='gray', opacity=0.8, lineWidth=0)
    patch.draw()


def instruction_images(text, images, width, delay, last=False, rect=False, double=True, duration=1):
    quick_exit()
    mywin.flip()
    instruct = visual.TextStim(mywin, text=text, wrapWidth=width,
                               alignHoriz='left', pos=(-(width/2), 100))
    wrap = 'When you are ready to begin, press SPACE' if last else 'Press SPACE to continue'
    color = (51, 204, 51) if last else 'white'
    bold = True if last else False
    space = visual.TextStim(mywin, text=wrap, wrapWidth=width, colorSpace='rgb255',
                            alignHoriz='center', color=color, pos=(0, -200), bold=bold)
    instruct.draw()
    space.draw()
    mywin.update()
    NoKey = True
    while NoKey:
        allKeys = event.getKeys()
        if len(allKeys) > 0:
            resp = allKeys[0]
            if resp == 'space':
                NoKey = False
    gen_image(images[0])
    if rect:
        gen_patch()
    instruct.draw()
    space.draw()
    mywin.update()
    instruct.draw()
    space.draw()
    core.wait(duration)
    mywin.flip()
    core.wait(1)
    if double:
        gen_image(images[1])
        instruct.draw()
        space.draw()
        mywin.update()
        core.wait(duration)
    mywin.flip()
    core.wait(delay)


def instruction_static(text, images, width, delay, last=False, double=True):
    quick_exit()
    mywin.flip()
    instruct = visual.TextStim(mywin, text=text, wrapWidth=width,
                               alignHoriz='left', pos=(-(width/2), 300))
    wrap = 'When you are ready to begin, press SPACE' if last else 'Press SPACE to continue'
    color = (51, 204, 51) if last else 'white'
    bold = True if last else False
    space = visual.TextStim(mywin, text=wrap, wrapWidth=width, colorSpace='rgb255',
                            alignHoriz='center', color=color, pos=(0, -300), bold=bold)
    if double:
        gen_image(images[0], position=[-150, 0])
        gen_image(images[1], position=[150, 0])
    else:
        gen_image(images[0], position=[0, -50], size=[790, 600])
    instruct.draw()
    space.draw()
    mywin.update()
    NoKey = True
    while NoKey:
        allKeys = event.getKeys()
        if len(allKeys) > 0:
            resp = allKeys[0]
            if resp == 'space':
                NoKey = False
    core.wait(delay)


def associative_inf_instruct_old():
    instruction_screen('The next task is a memory test. During the square-detection task, if you saw a certain image (A), it was ALWAYS followed by the same image (B). Each of those images was paired with a new image in the set you just studied. Image A was now paired with C, and B with D. In this next task, you will be provided with a particular image C, and will be asked to select the correct image D.',
                       800, 0.25)
    instruction_screen('In other words, you will be presented with a cue image, and three possible images as answers. You will need to make an inference and choose the image which was associated with the same initial pair as the cue.',
                       800, 0.25)
    instruction_images('To clarify, during the square detection task, a set of two images like these might always have been paired together...',
                       [103, 104], 800, 0.25)
    instruction_images('In the last task, you studied pairs where the first image was newly paired with an alternate image...', [103, 108], 800, 1)
    instruction_images('...and the second image was also newly paired with an alternate image.', [104, 111], 800, 0.25)
    instruction_static('For the upcoming test, you are presented with this image:', [108, 108], 800, 0.25, double=False)
    instruction_static('And the correct answer would be this image, because it was previously associated with the same intermediate pair:',
                       [111, 111], 800, 0.25, double=False)
    instruction_screen('On each trial, you will be presented with a cue image, and three possible answers below. Press the left arrow to choose the image on the lefthand side, the down arrow to choose the image in the middle, or the right arrow to choose the image on the right.',
                       800, 0.25)
    instruction_screen('If you have any questions about this, please ask the experimenter now.', 800, 2, last=True)


def associative_inf_instruct():
    instruction_screen('The next task is a memory test. You will be presented with a cue image, and three possible images as answers. You will need to choose the image which was associated with the same initial pair as the cue.',
                       800, 0.0)
    instruction_static('To clarify, during the square detection task, a set of two images like these might always have been paired together...',
                       ['Greyed.001']*2, 800, 0.0, double=False)
    instruction_static('In the last task, you studied pairs where the both of these images were newly paired with an alternate image...',
                       ['Greyed.002']*2, 800, 0.0, double=False)
    instruction_static('For this upcoming task, you will be presented with the cue image on the left, and the correct answer would be the image on the right, because it was previously associated with the same pair. That pair serves as an intermediate link.',
                       ['Greyed.003']*2, 800, 0.0, double=False)
    instruction_screen('On each trial, you will be presented with a cue image, and three possible answers below it. Press the left arrow to choose the image on the lefthand side, the down arrow to choose the image in the middle, or the right arrow to choose the image on the right.',
                       800, 0.0)
    instruction_screen('If you have any questions about this, please ask the experimenter now.', 800, 2, last=True)


def new_assoc_instruct():
    instruction_screen('The next task is a memory test. You will be presented with a cue image, and three possible images as answers. You will need to choose the image which was associated the cue.',
                       800, 0.25)
    instruction_screen('On each trial, you will be presented with a cue image, and three possible answers below it. Press the left arrow to choose the image on the lefthand side, the down arrow to choose the image in the middle, or the right arrow to choose the image on the right.',
                       800, 0.25)
    instruction_screen('If you have any questions about this, please ask the experimenter now.', 800, 2, last=True)


# Writes the data for a given trial
def write_data(task, run, trialn, pair, image, rect, answer, resp, resp_time, acc, sta):
    sta = sta.append({'Task': task, 'Run': run, 'Trial Number': trialn,
                      'Pair': pair, 'Item': image, 'Rect': rect,
                      'CorrResp': answer, 'Resp': resp, 'RT': resp_time,
                      'Acc': acc}, ignore_index=True)
    return sta


def write_newEncdata(task, run, trialn, pair, new_pair, lure_pair, correct,
                     cue, intrude, lure, newEnc):
    newEnc = newEnc.append({'Task': task, 'Run': run, 'Trial Number': trialn,
                            'CritPair': pair, 'SubPair': new_pair, 'LurePair': lure_pair,
                            'Cue': cue, 'Correct': correct, 'pmLure': intrude,
                            'uLure': lure}, ignore_index=True)
    return newEnc


def write_newRetdata(task, run, trialn, pair, new_pair, lure_pair, cue, correct, intrude, lure,
                     critpos, pmpos, upos, corresp, resp, resp_time, acc, data):
    data = data.append({'Task': task, 'Run': run, 'Trial Number': trialn, 'CritPair': pair, 'SubPair': new_pair,
                        'LurePair': lure_pair, 'Cue': cue, 'Correct': correct, 'pmLure': intrude, 'uLure': lure,
                        'CorrPos': critpos, 'pmPos': pmpos, 'uPos': upos, 'CorrResp': corresp, 'Resp': resp,
                        'RT': resp_time, 'Acc': acc}, ignore_index=True)
    return data



def write_numdata(task, run, trialn, items, critpair, critpos, critprobe,
                  fillprobe, corritem, corrresp, resp, resp_time, acc, num):
    num = num.append({'Task': task, 'Run': run, 'Trial Number': trialn, 'Items': items,
                      'CritPair': critpair, 'CritPos': critpos, 'CritProbe': critprobe,
                      'FillProbe': fillprobe, 'CorrItem': corritem, 'CorrResp': corrresp,
                      'Resp': resp, 'RT': resp_time, 'Acc': acc}, ignore_index=True)
    return num


# Runs an entire stat learning or pre/post trial
def trial_run(image, answer, trialn, rect, data, pair, run, rects=False, task='stat'):
    quick_exit()
    resp = 'miss'
    resp_time = None
    acc = None
    gen_image(image)
    if rects and rect:
        gen_patch()
    event.clearEvents()
    mywin.update()
    trialTimer.reset()
    noKey = True
    while trialTimer.getTime() < 1.0:
        allKeys=event.getKeys(timeStamped=trialTimer)
        if len(allKeys) > 0 and noKey:
            resp = allKeys[0][0]
            if resp == answer:
                acc = 1
            else:
                acc = 0
            resp_time = allKeys[0][1]
            noKey = False
    data = write_data(task, run, trialn, pair, image, rect, answer, resp, resp_time,
                      acc, data)
    return data


def newEnc_trial_run(cue, correct, intrude, lure, pair, new_pair, lure_pair,
                     trialn, run, data, enc_time):
    quick_exit()
    gen_image(cue)
    mywin.update()
    core.wait(enc_time)
    mywin.flip()
    core.wait(0.25)
    gen_image(correct)
    mywin.update()
    core.wait(enc_time)
    data = write_newEncdata('newEnc', run, trialn, pair, new_pair, lure_pair, cue,
                            correct, intrude, lure, data)
    return data


def newRet_trial_run(cue, correct, intrude, lure, pair, new_pair, lure_pair, trialn, run, data, ret_time):
    quick_exit()
    resp = 'miss'
    resp_time = None
    acc = None
    positions = ['left', 'down', 'right']
    pos_dict = dict(zip(positions, [[-150, -100], [-0, -100], [150, -100]]))
    random.shuffle(positions)
    gen_image(cue, position=[0, 100])
    gen_image(correct, position=pos_dict[positions[0]])
    gen_image(intrude, position=pos_dict[positions[1]])
    gen_image(lure, position=pos_dict[positions[2]])
    critpos = positions[0]
    pmpos = positions[1]
    upos = positions[2]
    answer = critpos
    mywin.update()
    event.clearEvents()
    trialTimer.reset()
    noKey = True
    while trialTimer.getTime() < ret_time:
        allKeys = event.getKeys(timeStamped=trialTimer)
        if len(allKeys) > 0 and noKey:
            resp = allKeys[0][0]
            if resp == answer:
                acc = 1
            else:
                acc = 0
            resp_time = allKeys[0][1]
            noKey = False
    data = write_newRetdata('newRet', run, trialn, pair, new_pair, lure_pair, cue,
                            correct, intrude, lure, critpos, pmpos, upos, answer,
                            resp, resp_time, acc, data)
    return data


def numer_trial_run(probe_stim, winner, items, trialn, run, num, rsvp_time=0.1, probe_lag=0.25, num_resp=1.5):
    quick_exit()
    resp = 'miss'
    resp_time = None
    acc = None
    if winner == 'targ':
        rsvp = items[0:2] * 6
        rsvp2 = items[2:4] * 4
    else:
        rsvp = items[2:4] * 6
        rsvp2 = items[0:2] * 4
    rsvp.extend(rsvp2)
    random.shuffle(rsvp)
    while any(i == j for i, j in zip(rsvp, rsvp[1:])):
        random.shuffle(rsvp)
    for image in rsvp:
        gen_image(image)
        mywin.update()
        core.wait(rsvp_time)
    mywin.flip()
    if np.random.randint(0, 2) == 0:
        # target left, filler right
        critpos = "left"
        gen_image(items[probe_stim], position=[-80, 0])
        gen_image(items[probe_stim + 2], position=[80, 0])
        answer = 'left' if winner == 'targ' else "right"
    else:
        # target right, filler left
        critpos = "right"
        gen_image(items[probe_stim], position=[80, 0])
        gen_image(items[probe_stim + 2], position=[-80, 0])
        answer = 'right' if winner == 'targ' else "left"
    core.wait(probe_lag)
    event.clearEvents()
    mywin.update()
    trialTimer.reset()
    noKey = True
    while trialTimer.getTime() < num_resp:
        allKeys = event.getKeys(timeStamped=trialTimer)
        if len(allKeys) > 0 and noKey:
            resp = allKeys[0][0]
            if resp == answer:
                acc = 1
            else:
                acc = 0
            resp_time = allKeys[0][1]
            noKey = False
    core.wait(1.0)
    num = write_numdata('numerosity', run, trialn, items, items[0:2], critpos,
                        items[probe_stim], items[probe_stim+2], winner, answer,
                        resp, resp_time, acc, num)
    return num


# Runs the ISI depending on the time input
def set_isi(time):
    mywin.flip()
    core.wait(time)


# run an associative inference block
def associative_inf(IDnum, repetitions, pairset, trialn, run, enc_isi=2, enc_time=1, ret_time=5, ret_isi=1, interval=60):
    infRet = pd.DataFrame(columns=['Task', 'Run', 'Trial Number', 'CritPair', 'SubPair',
                                   'LurePair', 'Cue', 'Correct', 'pmLure', 'uLure',
                                   'CorrPos', 'pmPos', 'uPos', 'CorrResp', 'Resp',
                                   'RT', 'Acc'])
    infEnc = pd.DataFrame(columns=['Task', 'Run', 'Trial Number', 'CritPair', 'SubPair',
                                   'LurePair', 'Cue', 'Correct', 'pmLure', 'uLure'])
    enc_list, test_list, vis_sim = gen_trial_new_assoc(repetitions, pairset)

    instruction_screen('In this next task, you will view pairs of images, one after the other. Please try to remember which images were paired together for a later memory test. You will see each pair of images '+ str(repetitions) + ' time(s).',
                       800, 0.25)
    instruction_screen('The first member of a pair will be presented for ' + str(enc_time) + ' second(s), followed closely by the second member of the pair. You do not need to press any buttons, just do your best to remember the pairs.',
                       800, 0.25)
    instruction_screen('Once all the pairs have been presented, there will be a ' + str(
        interval) + ' second waiting period before the test.',
                       800, 2, last=True)

    rawTimer.reset()
    for initial, correct, other, lure, fam_lure, pair, new_pair, lure_pair in enc_list:
        trialn += 1
        infEnc = newEnc_trial_run(initial, correct, fam_lure, lure,  pair, new_pair,
                                  lure_pair, trialn, run, infEnc, enc_time)
        set_isi(enc_isi)
    print(rawTimer.getTime())
    infEnc.to_csv("./Data/" + str(IDnum) + "_infEnc_" + str(run) + ".csv")
    trialn = 0
    set_isi(interval)

    associative_inf_instruct()

    rawTimer.reset()
    for _, cue, correct, lure, fam_lure, pair, new_pair, lure_pair in test_list:
        trialn += 1
        infRet = newRet_trial_run(cue, correct, fam_lure, lure, pair, new_pair,
                                  lure_pair, trialn, run, infRet, ret_time)
        set_isi(ret_isi)
    print(rawTimer.getTime())
    infRet['condition'] = vis_sim
    infRet.to_csv("./Data/" + str(IDnum) + "_infRet_" + str(run) + ".csv")


# Run new associate task
def new_associate(IDnum, repetitions, pairset, trialn, run, enc_isi=2, enc_time=1, ret_time=5, ret_isi=1, interval=60):
    newRet= pd.DataFrame(columns=['Task', 'Run', 'Trial Number', 'CritPair', 'SubPair',
                                'LurePair', 'Cue', 'Correct', 'pmLure', 'uLure',
                                'CorrPos', 'pmPos', 'uPos', 'CorrResp', 'Resp',
                                'RT', 'Acc'])
    newEnc = pd.DataFrame(columns=['Task', 'Run', 'Trial Number', 'CritPair', 'SubPair',
                                'LurePair', 'Cue', 'Correct', 'pmLure', 'uLure'])
    enc_list, test_list, vis_sim = gen_trial_new_assoc(repetitions, pairset)

    instruction_screen('In this next task, you will view pairs of images, one after the other. Please try to remember which images were paired together for a later memory test. You will see each pair of images '+ str(repetitions) + ' time(s).',
                       800, 0.25)
    instruction_screen('The first member of a pair will be presented for ' + str(enc_time) + ' second(s), followed closely by the second member of the pair. You do not need to press any buttons, just do your best to remember the pairs.',
                       800, 0.25)
    instruction_screen('Once all the pairs have been presented, there will be a ' + str(interval) + ' second waiting period before the test.',
                       800, 2, last=True)
    rawTimer.reset()
    for cue, correct, intrude, lure, _, pair, new_pair, lure_pair in enc_list:
        trialn += 1
        newEnc = newEnc_trial_run(cue, correct, intrude, lure, pair, new_pair,
                                  lure_pair, trialn, run, newEnc, enc_time)
        set_isi(enc_isi)
    print(rawTimer.getTime())
    newEnc.to_csv("./Data/" + str(IDnum) + "_newEnc_" + str(run) + ".csv")
    trialn = 0
    set_isi(interval)

    new_assoc_instruct()
    rawTimer.reset()
    for cue, correct, intrude, lure, _, pair, new_pair, lure_pair in test_list:
        trialn += 1
        newRet = newRet_trial_run(cue, correct, intrude, lure, pair, new_pair,
                                  lure_pair, trialn, run, newRet, ret_time)
        set_isi(ret_isi)
    print(rawTimer.getTime())
    newRet['condition'] = vis_sim
    newRet.to_csv("./Data/" + str(IDnum) + "_newRet_" + str(run) + ".csv")


# run a numerosity task block
def numerosity(IDnum, trials_per_pair, pairset, trialn, run, isi=0.5, rsvp_time=0.1, probe_lag=0.25):
    num = pd.DataFrame(columns=['Task', 'Run', 'Trial Number', 'Items', 'CritPair',
                                'CritPos', 'CritProbe', 'FillProbe', 'CorrItem',
                                'CorrResp', 'Resp', 'RT', 'Acc'])
    zipped_trials, vis_sim = gen_trial_num(trials_per_pair, pairset)

    instruction_screen('In this task, you will view a very rapid series of images. Following this, two images will appear. From these two images, you will need to select which of the two appeared most. If you think the image on the left appeared more than the one on the right, press the left arrow. If you think the image on the right appeared more, press the right arrow.',
                       800, 0.25)
    instruction_screen('If you have any questions, please ask the experimenter now.',
                       800, 2, last=True)

    rawTimer.reset()
    for probe_stim, winner, items in zipped_trials:
        trialn += 1
        num = numer_trial_run(probe_stim, winner, items, trialn, run,
                              num, rsvp_time, probe_lag)
        set_isi(isi)
    print(rawTimer.getTime())
    num['condition'] = vis_sim
    num.to_csv("./Data/" + str(IDnum) + "_Num_" + str(run) + ".csv")


# stat learning blocks
def stat_learning(IDnum, exposures, pairset, percent, trialn, run):
    sta = pd.DataFrame(columns=['Task', 'Run', 'Trial Number', 'Pair', 'Item',
                                'Rect', 'CorrResp', 'Resp', 'RT', 'Acc'])
    trial_list, pair_list = gen_trial_stat(exposures, pairset)
    isi_list = gen_isi(exposures)
    answers, rect_list = gen_cover(exposures, percent)
    trials = zip(trial_list, pair_list, isi_list, answers, rect_list)

    if run == 1:
        instruction_screen('In this square-detection task, you will view a series of images, one after the other. Occasionally, one of these images will have a small grey square over top of it. If this happens, press the LEFT arrow. Otherwise, do not press anything.',
                           800, 0.25)
        instruction_screen('If you have any questions, please ask the experimenter now.',
                           800, 2, last=True)
    else:
        instruction_screen('You will now complete the same task again.',
                           800, 2, last=True)

    set_isi(2.0)
    rawTimer.reset()
    for stim, pair, isi, answer, rect in trials[:]:
        trialn += 1
        sta = trial_run(stim, answer, trialn, rect, sta, pair, run, rects=True, task='stat')
        set_isi(isi)
    print(rawTimer.getTime())
    sta.to_csv("./Data/" + str(IDnum) + "_Stat_" + str(run) + ".csv")


# pre and post perceptual templating
def pre_post(IDnum, exposures, pairset, percent, trialn, run):
    prepost = pd.DataFrame(columns=['Task', 'Run', 'Trial Number', 'Pair', 'Item',
                                'Rect', 'CorrResp', 'Resp', 'RT', 'Acc'])
    trial_list, pair_list = gen_trial_prepost(exposures, pairset)
    isi_list = gen_isi(exposures)
    answers, rect_list = gen_cover(exposures, percent)
    trials = zip(trial_list, pair_list, isi_list, answers, rect_list)

    if run == 1:
        instruction_screen('In this square-detection task, you will view a series of images, one after the other. Occasionally, one of these images will have a small grey square over top of it. If this happens, press the LEFT arrow. Otherwise, do not press anything.',
                           800, 0.25)
        instruction_screen('If you have any questions, please ask the experimenter now.',
                           800, 2, last=True)
    else:
        instruction_screen('You will now complete the same task again.',
                           800, 2, last=True)

    set_isi(2.0)
    for stim, pair, isi, answer, rect in trials[:]:
        trialn += 1
        prepost = trial_run(stim, answer, trialn, rect, prepost, pair, run, rects=True, task='prepost')
        set_isi(isi)
    prepost.to_csv("./Data/" + str(IDnum) + "_PrePost_" + str(run) + ".csv")


# run a full experiment
def full_exp(pairset, n_prepost, n_stat, n_differint, differint, num_per=5, t_p_p=10, rect_per=10, reps=1,
             pres=[False], IDnum='test'):
    prenum = 0
    for task, pre in zip(differint, pres):
        if pre:
            for i in range(n_differint):
                if task == 'numerosity':
                    numerosity(IDnum, trials_per_pair=t_p_p, pairset=pairset, trialn=0, run=i+1)
                elif task == 'new_associates':
                    new_associate(IDnum, repetitions=reps, pairset=pairset, trialn=0, run=i+1)
                elif task == 'associative_inf':
                    associative_inf(IDnum, repetitions=reps, pairset=pairset, trialn=0, run=i+1)
                prenum += 1
    for i in range(n_prepost):
        pre_post(IDnum, exposures=num_per, pairset=pairset, percent=rect_per, trialn=0, run=i+1)
    for i in range(n_stat):
        stat_learning(IDnum, exposures=num_per, pairset=pairset, percent=rect_per, trialn=0, run=i+1)
    for i in range(n_prepost):
        pre_post(IDnum, exposures=num_per, pairset=pairset, percent=rect_per, trialn=0, run=i+1+n_prepost)
    for task in differint:
        for i in range(n_differint):
            if task == 'numerosity':
                numerosity(IDnum, trials_per_pair=t_p_p, pairset=pairset, trialn=0, run=i+prenum+1, rsvp_time=0.1)
            elif task == 'new_associates':
                new_associate(IDnum, repetitions=reps, pairset=pairset, trialn=0, run=i+prenum+1)
            elif task == 'associative_inf':
                associative_inf(IDnum, repetitions=reps, pairset=pairset, trialn=0, run=i+prenum+1)
    instruction_screen('You have finished! Please find the experimenter.', 800, 5, last=False, done=True)



full_exp(basepairs, templates, stats, outcomes, tasks, num_per=exposures, t_p_p=nums, rect_per=rectangles,
         pres=befaft, reps=encodings, IDnum=subid)

mywin.close()
core.quit()
