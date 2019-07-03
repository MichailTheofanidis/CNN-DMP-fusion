#!/usr/bin/env python

import matplotlib.pyplot as plt


def phase(s, title='Phase'):
    figure = plt.figure()
    plt.plot(s)
    figure.suptitle(title)
    return figure


def gaussian(s, psv, w, title):
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    figure = plt.figure()
    figure.suptitle(title)
    for i in range(w.shape[1]):
        plt.subplot(w.shape[1], 1, i + 1)
        for j in range(psv.shape[0]):
            plt.plot(s, psv[j, :] * w[j, i], color=colors[j % len(colors)])
    return figure


def plot_position(t, q, title='Positions'):

    figure = plt.figure()
    figure.suptitle(title)
    counter = 0
    for i in range(q.shape[1]):
        plt.subplot(q.shape[1], 1, i + 1)
        plt.plot(t, q[:, i])
        counter += 1
    return figure


def show_all():
    plt.show()
