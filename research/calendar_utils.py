#!/usr/bin/env python3
"""Календарная атрибуция меток времени. Одно определение на весь репозиторий.

ЧТО ИСПРАВЛЯЕТ
==============
В расчётах использовалось `year = день // 365 + 1970`. Это деление на 365, а не
календарное преобразование: високосные дни не учитываются, и граница года
уползает примерно на 0,2425 суток в год. К 2020-м смещение достигает почти двух
недель, из-за чего конец декабря относился к следующему году.

Здесь год, квартал и дата считаются настоящим преобразованием времени. Часовой
пояс атрибуции объявлен: **America/New_York**, потому что торговые сутки
репозитория привязаны к 18:00 ET. Метки ленты хранятся как UTC-наносекунды.

Разделение выборок по дате в коде поиска использовало точную дату и этой ошибкой
не затронуто; затронуты только годовые и квартальные ПОДПИСИ.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TZ = 'America/New_York'


def eastern(ts_ns):
    """UTC-наносекунды → локальное время торговой атрибуции."""
    return pd.to_datetime(np.asarray(ts_ns, dtype='int64'), utc=True).tz_convert(TZ)


def year_of(ts_ns):
    return eastern(ts_ns).year.to_numpy()


def quarter_of(ts_ns):
    e = eastern(ts_ns)
    return (e.year.to_numpy() * 4 + (e.quarter.to_numpy() - 1))


def quarter_label(key):
    return f'{int(key) // 4}Q{int(key) % 4 + 1}'


def date_key(ts_ns):
    """Целочисленный ключ локальной даты; годится для группировки по дням."""
    e = eastern(ts_ns)
    return (e.year.to_numpy() * 10000 + e.month.to_numpy() * 100
            + e.day.to_numpy())


def minute_of_day(ts_ns):
    e = eastern(ts_ns)
    return e.hour.to_numpy() * 60 + e.minute.to_numpy()


def naive_year_offset(ts_ns):
    """Насколько прежняя подпись расходилась с календарной. Для отчёта."""
    naive = (np.asarray(ts_ns, dtype='int64') // 60_000_000_000) // 1440
    naive = naive // int(365.2425) + 1970
    true = year_of(ts_ns)
    return {'mislabelled': int((naive != true).sum()),
            'total': int(len(true)),
            'share': float((naive != true).mean())}
