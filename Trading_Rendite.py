#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 29 12:07:20 2021

@author: leon
"""

konto = [1040]
depot = [0]
wert = [61.18]

N = 10

Kosten = 0

Rel = 0

Pos = 1
Neg = 0.448
Up = 1.02
Down = 0.99

gesamtkosten = 2 * N * Kosten

for I in range(N):
    
    depot.append((konto[2*I] - Kosten )/ wert[2*I])
    konto.append(0)
    
    if Rel == 0:
        wert.append(round(wert[2*I] + Pos,3))
    else:
        wert.append(round(wert[2*I] * Up,3))
    
    if konto[2*I] < 0:
        raise ValueError('Nicht genug Geld')
    
    
    konto.append(round(depot[2*I+1] * wert[2*I+1] - Kosten,2))
    depot.append(0)
    
    if Rel == 0:
        wert.append(round(wert[2*I+1] - Neg,3))
    else:
        wert.append(round(wert[2*I+1] * Down,3))
    
    
    
print('Kontoguthaben: ',konto[-1])
print('Gesamtkosten: ',gesamtkosten)
print('Endwert Instrument: ', wert[-1])