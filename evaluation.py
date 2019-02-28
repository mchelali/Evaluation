#-*-coding: utf-8 -*-

import numpy as np
import skimage.io as sk

def precision(y_label, y_predicted):
    """
    :param y_result: vecteur de la classif a evaluer
    :param y_label: vecteur de la verite terrain
    :return: liste de la precision pour chaque classe
    """

    ######################################################################################
    ######################################################################################
    #                                                                                    #
    #                   c = nb de doc correctement attribues a la classe i               #
    #       Precision = --------------------------------------------------------         #
    #                     s =  nb de documents appartenant a la classe i                 #
    #                                          -----------                               #
    #                                                                                    #
    ######################################################################################
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
        if s!=0:
            precision[i] = (c/s)

    return precision


def rapelle(y_label, y_predicted):
    """
    :param y_result: vecteur de la classif a evaluer
    :param y_label: vecteur de la verite terrain
    :return: liste du rapelle pour chaque classe
    """

    ######################################################################################
    ######################################################################################
    #                                                                                    #
    #                     c =  nb de doc correctement attribues a la classe i            #
    #       Precision = --------------------------------------------------------         #
    #                      s =   nb de documents attribues a la classe i                 #
    #                                            ---------                               #
    #                                                                                    #
    ######################################################################################
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
        rapelle[i] = (c / s) if s!=0 else 0

    return rapelle

def f1_score(precision, rapelle):
    #########################################################
    #########################################################
    #                                                       #
    # * Le calcul du F1-score ce fait pour chaque classe i  #
    #   --------------------------------------------------  #
    #                                                       #
    #                         Precision * Rapelle           #
    #     F1-Score =  2 * -------------------------         #
    #                         Precision + Rapelle           #
    #                                                       #
    #########################################################
    #########################################################
    f1score = [0 for i in range(len(precision))]

    for i in range(len(precision)):
        if (precision[i]+rapelle[i]) != 0:
            f1score[i] = 2 * ( (precision[i]*rapelle[i])/(precision[i]+rapelle[i]) )

    return f1score

def support(y_label):
    nbr_class = len(set(y_label))
    sup = [0 for i in range(nbr_class)]
    for i in range(nbr_class):
        for j in range(len(y_label)):
            if y_label[j] == i:
                sup[i] += 1

    return sup

def rapport_classif(y_true, y_predicted):
    """
            Cette fonction permet de faire un rapport sur une classification faite en comparant le résultat
        obtenu avec une labélisation existante (vérité-terrain); dans ce rapport en calculant le rappel, la précision
        et la F-mesure (la moyenne harmonique entre le rappel et la précision
        :param y_true:
        :param y_predicted:
    :return: rap: chaine de caractere
    """
    p = precision(y_true, y_predicted)
    r = rapelle(y_true, y_predicted)
    f1 = f1_score(p, r)
    s = support(y_true)
    rap = ""
    rap += "\t\tprecision\trapel\tf1-score\tsupport\n"
    for i in range(len(p)):
        #print i, "\t | \t", p[i], "\t | \t", r[i], "\t | \t", f1[i]
        rap += " {:6d}\t|\t{:.2f}\t|\t{:.2f}\t|\t{:.2f}|\t{:d}\n".format(i, p[i], r[i], f1[i], s[i])
    p_globale = 0.
    r_globale = 0.
    f_globale = 0.
    for i in range(len(p)):
        p_globale += p[i]*s[i]
        r_globale += r[i]*s[i]

    if sum(s)!=0:
        p_globale = p_globale / sum(s)
        r_globale = r_globale / sum(s)
        f_globale = 2 * ((p_globale*r_globale)/(p_globale+r_globale))
    else:
        p_globale = 1
        r_globale = 1
        f_globale = 1

    #print "\nglobale \t | \t {:.4f} \t | \t {:.4f} \t | \t {:.4f} | \t {:d}".format(sum(p)/len(p),  sum(r)/len(r),  sum(f1)/len(f1), sum(s))
    rap += "\nwheigthed\t|\t{:.4f}\t|\t{:.4f}\t|\t{:.4f}|\t{:d}\n".format(p_globale,  r_globale,  f_globale, sum(s))

    return rap


def confMatrix(y_true, y_predicted, y_true_label=()):
    """
            Faire la matrice de confusion
        :param y_true: vecteur de la verriter terrain
        :param y_predicted: vecteur contenant la labelisation de la classification
        :param y_true_label:
        :return: 2 var are returned;
                matConf ==> la matrice de confusion
                rap ==> la matrice de confusion en format text pour pouvoir l'enregistrer sur un fichier
    """

    nbrClass = len(set(y_true))
    nbrCluster = len(set(y_predicted))

    if len(y_true_label)==0:
        y_true_label = range(nbrClass)

    matConf = np.zeros((nbrClass, nbrCluster), dtype=int)

    for i in range(len(y_predicted)):
        matConf[y_true[i], y_predicted[i]] += 1

    i = 0
    rap = " /* Matrice de confusion */ \n  -----------------------  \n"
    for row in matConf:
        rap += str(y_true_label[i]) + "\t"
        for col in row:
            rap = rap + str(col) + "\t"
        rap+="\n"
        i+=1

    return matConf, rap

def clusterAffectation(y_true, y_predicted, y_true_label=()):
    """
        Cette fonction permet d'affecter les clusers aux class correspondant tout en maximisant le taux de reconnaissance
    :param y_true: vecteur de la vérite terrain
    :param y_predicted: vecteur obtenue par un clustering ou classification
    :param y_true_label:  vecteur contanant le nom des classes
    :return: assigned_label: la nouvelle labelisation qui est affecté par rapport au meilleur taux de reconnaissance
    """

    nbrClass = len(set(y_true))
    nbrCluster = len(set(y_predicted))

    if len(y_true_label) == 0:
        y_true_label = range(nbrClass)

    matConf, rap = confMatrix(y_true, y_predicted, y_true_label)

    print (" Recherche de la bonne combine ")
    tauxRec = 0.
    cluster_set = []
    i = 1
    # Attribution de chaque cluster a son label de VT ( selon le nbr d'apparition  dans la classe)
    while (i < (nbrClass) ** nbrCluster - 1):
        comb_tmp = np.base_repr(i, nbrClass)
        # if len(comb_tmp) != n_clus:
        comb = [int(a) for a in np.base_repr(i, nbrClass, nbrCluster - len(comb_tmp))]
        tr = 0.
        for j in range(nbrCluster):
            tr += matConf[comb[j], j]
        tr = tr / sum(sum(matConf))
        if tr > 0.5 and len(set(comb))==nbrClass:
            print (comb, "RR = ", tr)
        if tr > tauxRec and len(set(comb))==nbrClass:
            tauxRec = tr
            cluster_set = comb
        i += 1

    comb_ = ""
    for c in cluster_set:
        comb_ += str(c) + " "

    print ("The best comb is ", cluster_set, "RR=", tauxRec)
    comb_ += " this is the best combinision with TR = " + str(tauxRec)

    assigned_label = []
    for pix in y_predicted:
        assigned_label.append(cluster_set[pix])

    return assigned_label, cluster_set, comb_

def groundTruth_preprocessing(path_vt, ignore_class = []):
    """

    :param path_vt:
    :param ignore_class:
    :return: vt: vecteur de la nouvelle labelisation des classes extraite à partir de l'image en entrer coloré
            points : Matrice binaire correspond au points a prendre en considiration pour tte autre post-traitement
            color_for_result: liste des couleurs des classes de la vt
    """

    y_true = sk.imread(path_vt)[:, :, 0:3]
    l, c, d = y_true.shape

    colors = list()
    for row in y_true:
        for pix in row:
            colors.append(pix)
    colors = np.array(list(set(tuple(r) for r in colors)))

    print("liste des couleurs : ", colors)
    y_true = np.mean(y_true, axis=2).astype(np.int)

    ng = list(np.mean(colors, axis=1).astype(np.int))
    ng_ = list(np.mean(colors, axis=1).astype(np.int))

    print ("Couleur acctuel de la vt sont : ", ng)

    color_for_result = []
    for n in range(len(ng_)):
        if ng_[n] in ignore_class:
            ng.remove(ng_[n])
        else:
            color_for_result.append(colors[n, :])

    """for a in range(len(ignore_class)):
        if ignore_class[a] in ng:
            ng.remove(ignore_class[a])
        else:
            color_for_result.append(colors[a, :])"""

    ng = (list(ng))
    nbrClass = len(ng)

    print("Liste des NG : ", ng)
    print("liste des couleurs : ", color_for_result)

    y_true2 = np.zeros_like(y_true, dtype=np.int)
    for i in range(nbrClass):
        y_true2[y_true == ng[i]] = i + 1

    #print(y_true2.shape)
    #print(set(y_true2.ravel()))

    point = np.zeros_like(y_true2)
    point[y_true2 != 0] = 1

    y_true2 = y_true2 - 1 # pour remettre les indices des classes à partir de 0
    vt = y_true2[point != 0].ravel()
    #print point.shape
    #print len(point[point != 0].ravel())

    return vt, point, color_for_result, y_true2


def EQM(signal_original, signal_traiter):
    """
               1
        EQM = --- sum [ (S_0 - S_r)^2 ]
               m

               S_0 : signal_original
               S_r : signal_traiter
               m : lengeur du signal

        :param signal_original:  Signal original
        :param signal_traiter: Signal traiter
        -------------------------------------------------
        :return: l’erreur quadratique moyenne (EQM)
    """

    if len(signal_original) != len(signal_traiter):
        raise ValueError("Les deux signaux n'ont pas la meme taille !!!")


    m = float(len(signal_original))

    # convertir les deux signaux en numpy array de type float
    signal_original = np.array(signal_original).astype(np.float)
    signal_traiter = np.array(signal_traiter).astype(np.float)

    eqm = np.sum(np.power(signal_original - signal_traiter, 2)) / m

    return eqm

def EQM_SITS(SITS_orginal, SITS_traiter):
    """
            Cette fonction permet de calculer le taux d'erreur estimé entre les deux series d'images
            en se basant sur le EQM

        :param SITS_orginal:  Satellite Image Time Serie original
        :param SITS_modified: Satellite Image Time Serie modifier ou traiter
        ---------------------------------------------------------------------------------------------
        :return: eqm moyen de tous les pixels temporels; Qui est le taux d'erreur estimé entre les 2 series d'images
    """
    l_0, c_0, d_0 = SITS_orginal.shape
    l_1, c_1, d_1 = SITS_traiter.shape

    if l_0 != l_1 or c_0 != c_1 or d_0 != d_1:
        raise ValueError("Les deux SITS n'ont pas les memes dimentions")

    SITS_orginal = SITS_orginal.reshape((-1, d_0))
    SITS_traiter = SITS_traiter.reshape((-1, d_1))

    eqm = 0.
    for i in range(l_0 * c_0):
        eqm += EQM(SITS_orginal[i, :], SITS_traiter[i, :])

    eqm /= (l_0 * c_0)

    return eqm

def EAM(signal_original, signal_traiter):
    """
               1
        EQM = --- sum [ (S_0 - S_r)^2 ]
               m

               S_0 : signal_original
               S_r : signal_traiter
               m : lengeur du signal

        :param signal_original:  Signal original
        :param signal_traiter: Signal traiter
        -------------------------------------------------
        :return: l’erreur quadratique moyenne (EQM)
    """

    if len(signal_original) != len(signal_traiter):
        raise ValueError("Les deux signaux n'ont pas la meme taille !!!")


    m = float(len(signal_original))

    # convertir les deux signaux en numpy array de type float
    signal_original = np.array(signal_original).astype(np.float)
    signal_traiter = np.array(signal_traiter).astype(np.float)

    eqm = np.sum(np.abs(signal_original - signal_traiter)) / m

    return eqm

def EAM_SITS(SITS_orginal, SITS_traiter):
    """
            Cette fonction permet de calculer le taux d'erreur estimé entre les deux series d'images
            en se basant sur le EQM

        :param SITS_orginal:  Satellite Image Time Serie original
        :param SITS_modified: Satellite Image Time Serie modifier ou traiter
        ---------------------------------------------------------------------------------------------
        :return: eqm moyen de tous les pixels temporels; Qui est le taux d'erreur estimé entre les 2 series d'images
    """
    l_0, c_0, d_0 = SITS_orginal.shape
    l_1, c_1, d_1 = SITS_traiter.shape

    if l_0 != l_1 or c_0 != c_1 or d_0 != d_1:
        raise ValueError("Les deux SITS n'ont pas les memes dimentions")

    SITS_orginal = SITS_orginal.reshape((-1, d_0))
    SITS_traiter = SITS_traiter.reshape((-1, d_1))

    eqm = 0.
    for i in range(l_0 * c_0):
        eqm += EAM(SITS_orginal[i, :], SITS_traiter[i, :])

    eqm /= (l_0 * c_0)

    return eqm

if __name__=="__main__":
    print (" -----* START *----- ")

    y_true      = [0, 1, 0, 1, 2, 1, 2, 0, 1, 2, 0, 0, 2, 2, 0, 2, 2, 1, 1, 2]
    y_predicted = [0, 1, 0, 2, 2, 2, 2, 0, 1, 2, 0, 0, 2, 2, 0, 2, 2, 1, 1, 2]

    print (y_true)
    print (y_predicted)

    p = (precision(y_true, y_predicted))
    r = rapelle(y_true, y_predicted)
    f1 = f1_score(p, r)

    print ("P = ", p)
    print ("R = ", r)
    print ("F1 = ", f1)
    print (rapport_classif(y_true, y_predicted))

    print ("EQM = ", EQM(y_true, y_predicted))


    #print classification_report(y_true, y_predicted)
