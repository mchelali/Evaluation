import numpy as np
import random
from sklearn.metrics import classification_report

def precision(y_label, y_predicted):
    """
    :param y_result: vecteur de la classif a evaluer
    :param y_label: vecteur de la verite terrain
    :return: liste de la precision pour chaque classe
    """

    ######################################################################################
    #                                                                                    #
    #                   c = nb de doc correctement attribues a la classe i               #
    #       Precision = --------------------------------------------------------         #
    #                     s =  nb de documents appartenant a la classe i                 #
    #                                          -----------                               #
    #                                                                                    #
    ######################################################################################

    nbr_class = len(set(y_label))
    precision = [0 for i in range(nbr_class)]
    for i in range(nbr_class):
        c = 0.
        s = 0.
        for j in range(len(y_label)):
            if y_label[j] == i:
                if y_predicted[j] == i:
                    c += 1.
            if y_predicted[j] == i:
                s += 1.
        precision[i] = (c/s)

    return precision


def rapelle(y_label, y_predicted):
    """
    :param y_result: vecteur de la classif a evaluer
    :param y_label: vecteur de la verite terrain
    :return: liste du rapelle pour chaque classe
    """

    ######################################################################################
    #                                                                                    #
    #                     c =  nb de doc correctement attribues a la classe i            #
    #       Precision = --------------------------------------------------------         #
    #                      s =   nb de documents attribues a la classe i                 #
    #                                            ---------                               #
    #                                                                                    #
    ######################################################################################

    nbr_class = len(set(y_label))
    rapelle = [0 for i in range(nbr_class)]
    for i in range(nbr_class):
        c = 0.
        s = 0.
        for j in range(len(y_label)):
            if y_label[j] == i:
                if y_predicted[j] == i:
                    c += 1.
                s += 1.
        rapelle[i] = (c / s)

    return rapelle

def f1_score(precision, rapelle):
    #########################################################
    #                                                       #
    #                         Precision * Rapelle           #
    #     F1-Score =  2 * -------------------------         #
    #                         Precision + Rapelle           #
    #                                                       #
    #########################################################
    f1score = [0 for i in range(len(precision))]

    for i in range(len(precision)):
        f1score[i] = 2 * ( (precision[i]*rapelle[i])/(precision[i]+rapelle[i]) )

    return f1score

if __name__=="__main__":
    print " -----* START *----- "

    y_true      = [0, 1, 0, 1, 2, 1, 2, 0, 1, 2, 0, 0, 2, 2, 0, 2, 2, 1, 1, 2]
    y_predicted = [0, 1, 0, 0, 2, 2, 2, 0, 1, 2, 0, 0, 2, 2, 1, 0, 0, 1, 1, 0]

    print y_true
    print y_predicted

    p = precision(y_true, y_predicted)
    r = rapelle(y_true, y_predicted)
    f1 = f1_score(p, r)

    print "P = ", p
    print "R = ", r
    print "F1 = ", f1

    print "p_globale = ", sum(p)/len(p)
    print "r_globale = ", sum(r)/len(r)
    print "f1_globale = ", sum(f1)/len(f1)

    print classification_report(y_true, y_predicted)


    print " ------* END *------ "
