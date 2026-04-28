#!/usr/bin/env python
import os
import numpy as np
import pandas
import lhapdf
import time

try:
    from Collins_fit.TMDs.udis    import UDIS
    from Collins_fit.TMDs.pdis    import PDIS
    from Collins_fit.input_params import inputparams
    from Collins_fit.data.reader import load_data
    from Collins_fit.kits.kits   import check_path
    from Collins_fit.kits.share  import share

except:
    import sys
    sys.path.append("../")
    sys.path.append("../..")
    from TMDs.udis    import UDIS
    from TMDs.pdis    import PDIS
    from input_params import inputparams
    from data.reader import load_data
    from kits.kits   import check_path
    from kits.share  import share


def SIDIS():
    for index in list(share['data']['SIDIS']):
        results = []
        for i in range(len(share['data']['SIDIS'][index]['value'])):
            x   = share['data']['SIDIS'][index]['x'  ][i]
            y   = share['data']['SIDIS'][index]['y'  ][i]
            z   = share['data']['SIDIS'][index]['z'  ][i]
            pht = share['data']['SIDIS'][index]['pht'][i]
            Q2  = share['data']['SIDIS'][index]['Q2' ][i]
            Q   = np.sqrt(Q2)
            den = share['DIS' ].dsigmadxdydzd2PhT_p_SIDIS(pht, x, y, z, Q)
            num = share['PDIS'].dsigmadxdydzd2PhT_p_SIDIS(pht, x, y, z, Q)
            results.append(num / den)

        results = np.array(results)
        share['data']['SIDIS'][index]['theory'] = results
        dataframe = pandas.DataFrame(data = share['data']['SIDIS'][index])
        name = './results/' + 'SIDIS' + '/'
        check_path(name)
        name += str(index) + '.xlsx'
        dataframe.to_excel(name, index = False)

    return

if __name__ == '__main__':
    try:
        from Collins_fit.input_params import inputparams
    except:
        from input_params import inputparams

    scheme = 'MSbar'
    scheme = 'JCC'
    scheme = 'CSS'
    share['scheme'] = scheme

    ## setup LHAPDF
    PDF_name = f'CxCT14nlo-{scheme}'
    FF_name  = f'CxDSS14PI-{scheme}'
    share['LHAPDF'] = {}
    share['LHAPDF']['PDFs'] = lhapdf.mkPDF(PDF_name, 1)
    share['LHAPDF']['FFs' ] = lhapdf.mkPDF(FF_name , 1)

    ## load data
    share['data'] = {}
    share['data']['SIDIS'] = {}
    indices = [140801]
    for index in indices:
        share['data']['SIDIS'][index] = load_data('SIDIS', index)

    parameters = []
    parameters.append(inputparams['h1']['KPSY']['N']['uu'])
    parameters.append(inputparams['h1']['KPSY']['a']['uu'])
    parameters.append(inputparams['h1']['KPSY']['b']['uu'])
    parameters.append(inputparams['h1']['KPSY']['N']['dd'])
    parameters.append(inputparams['h1']['KPSY']['a']['dd'])
    parameters.append(inputparams['h1']['KPSY']['b']['dd'])
    parameters.append(inputparams['h1']['KPSY']['N']['ss'])
    parameters.append(inputparams['h1']['KPSY']['a']['ss'])
    parameters.append(inputparams['h1']['KPSY']['b']['ss'])
    parameters.append(inputparams['h1']['KPSY']['N']['sb'])
    parameters.append(inputparams['h1']['KPSY']['a']['sb'])
    parameters.append(inputparams['h1']['KPSY']['b']['sb'])
    parameters.append(inputparams['h1']['KPSY']['N']['db'])
    parameters.append(inputparams['h1']['KPSY']['a']['db'])
    parameters.append(inputparams['h1']['KPSY']['b']['db'])
    parameters.append(inputparams['h1']['KPSY']['N']['ub'])
    parameters.append(inputparams['h1']['KPSY']['a']['ub'])
    parameters.append(inputparams['h1']['KPSY']['b']['ub'])

    parameters.append(inputparams['H3']['KPSY']['N']['fav'])
    parameters.append(inputparams['H3']['KPSY']['N']['unf'])
    parameters.append(inputparams['H3']['KPSY']['a']['fav'])
    parameters.append(inputparams['H3']['KPSY']['a']['unf'])
    parameters.append(inputparams['H3']['KPSY']['b']['fav'])
    parameters.append(inputparams['H3']['KPSY']['b']['unf'])

    loops = 1
    logs  = 3
    share['DIS'] = UDIS(loops, logs)
    loops = 1
    logs  = 2
    share['PDIS'] = PDIS(parameters, loops = loops, logs = logs)

    t_0 = time.time()
    SIDIS()
    et  = time.time() - t_0
    et /= 60.0
    print(f'time used: {et:.2f} minutes')

