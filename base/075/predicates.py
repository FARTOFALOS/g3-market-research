"""075: the candle vocabulary of the node, as named conditions.

Every predicate is a statement about candles that are already closed at minute k,
about the zone of the scene, or about the clock. None of them can see minute k+1
or later. Their names are the names used in the card.
"""
import numpy as np
import pandas as pd


def build(f):
    P = {}
    P['body with the push']        = (f.body > 0).to_numpy()
    P['closes in its last fifth']  = (f.close_in_range >= .8).to_numpy()
    P['range above its neighbour'] = (f.range_pts > f.range_pts.shift(0)).to_numpy() & True  # placeholder
    P['takes the previous extreme'] = (f.pushes_prev == 1).to_numpy()
    P['engulfs the previous body'] = (f.engulf_prev == 1).to_numpy()
    P['inside the previous minute'] = (f.inside_prev == 1).to_numpy()
    P['previous minute pushed too'] = (f.prev_pushes == 1).to_numpy()
    P['short visit to the zone']    = (f.zlen <= 2).to_numpy()
    P['long visit to the zone']     = (f.zlen >= 5).to_numpy()
    P['visit went a width deep']    = (f.zdepth >= 1.).to_numpy()
    P['visit only grazed the zone'] = (f.zdepth <= .2).to_numpy()
    P['first departure of the film'] = f['first'].to_numpy().astype(bool)
    P['film already saw the far side'] = (f.been_far == 1).to_numpy()
    P['third visit or later']       = (f.n_prior_Z >= 3).to_numpy()
    P['early from T0 (k<=10)']      = (f.k <= 10).to_numpy()
    P['late from T0 (k>=30)']       = (f.k >= 30).to_numpy()
    P['stop wider than the minute range'] = (f.risk > f.range_pts).to_numpy()
    P['stop at least one ATR30']    = (f.risk >= f.atr30).to_numpy()
    P['stop under 0.6 ATR30']       = (f.risk < .6 * f.atr30).to_numpy()
    P['zone wider than ATR30']      = (f.w_atr >= 1.).to_numpy()
    P['zone under half an ATR30']   = (f.w_atr < .5).to_numpy()
    P['closes a full width clear']  = (f.close_beyond >= 1.).to_numpy()
    P['closes just past the border'] = (f.close_beyond < .25).to_numpy()
    P['regular session']            = (f.rth == 1).to_numpy()
    P['first half hour of the session'] = ((f.minute_of_day >= 570) & (f.minute_of_day < 600)).to_numpy()
    P['13:00-14:00 New York']       = ((f.minute_of_day >= 780) & (f.minute_of_day < 840)).to_numpy()
    P['away from the zone (ZE)']    = (f.kind == 'ZE').to_numpy()
    P['through the zone (ZF)']      = (f.kind == 'ZF').to_numpy()
    P.pop('range above its neighbour')
    return P
