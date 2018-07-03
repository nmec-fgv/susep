############################################################################
## Captura e transformação de dados da base postgres para arquivos pickle ##
############################################################################

import os
import pickle
import psycopg2
import numpy as np
import re
import pdb
from datetime import datetime
from dateutil.relativedelta import relativedelta

# Data directory:
data_dir = '/home/pgsqldata/Susep/'

# IPCA for severity adjustment, below is hardcoded jan-2008 to dez-2016, with jan-2008 as base value 1.0

dates = ['2008-01', '2008-02', '2008-03', '2008-04', '2008-05', '2008-06', '2008-07', '2008-08', '2008-09', '2008-10', '2008-11', '2008-12', '2009-01', '2009-02', '2009-03', '2009-04', '2009-05', '2009-06', '2009-07', '2009-08', '2009-09', '2009-10', '2009-11', '2009-12', '2010-01', '2010-02', '2010-03', '2010-04', '2010-05', '2010-06', '2010-07', '2010-08', '2010-09', '2010-10', '2010-11', '2010-12', '2011-01', '2011-02', '2011-03', '2011-04', '2011-05', '2011-06', '2011-07', '2011-08', '2011-09', '2011-10', '2011-11', '2011-12', '2012-01', '2012-02', '2012-03', '2012-04', '2012-05', '2012-06', '2012-07', '2012-08', '2012-09', '2012-10', '2012-11', '2012-12', '2013-01', '2013-02', '2013-03', '2013-04', '2013-05', '2013-06', '2013-07', '2013-08', '2013-09', '2013-10', '2013-11', '2013-12', '2014-01', '2014-02', '2014-03', '2014-04', '2014-05', '2014-06', '2014-07', '2014-08', '2014-09', '2014-10', '2014-11', '2014-12', '2015-01', '2015-02', '2015-03', '2015-04', '2015-05', '2015-06', '2015-07', '2015-08', '2015-09', '2015-10', '2015-11', '2015-12', '2016-01', '2016-02', '2016-03', '2016-04', '2016-05', '2016-06', '2016-07', '2016-08', '2016-09', '2016-10', '2016-11', '2016-12']

factors = [1.00000000, 1.00540000, 1.01032646, 1.01517603, 1.02075950, 1.02882350, 1.03643679, 1.04192990, 1.04484731, 1.04756391, 1.05227795, 1.05606615, 1.05902313, 1.06410645, 1.06995903, 1.07209895, 1.07724502, 1.08230808, 1.08620438, 1.08881127, 1.09044449, 1.09306156, 1.09612213, 1.10061623, 1.10468851, 1.11297368, 1.12165487, 1.12748748, 1.13391415, 1.13878999, 1.13878999, 1.13890386, 1.13935943, 1.14448654, 1.15307019, 1.16264067, 1.16996531, 1.17967602, 1.18911343, 1.19850743, 1.20773593, 1.21341229, 1.21523241, 1.21717678, 1.22168034, 1.22815524, 1.23343631, 1.23985018, 1.24604943, 1.25302731, 1.25866593, 1.26130913, 1.26938151, 1.27395128, 1.27497044, 1.28045281, 1.28570267, 1.29303118, 1.30066006, 1.30846402, 1.31880089, 1.33014257, 1.33812343, 1.34441261, 1.35180688, 1.35680856, 1.36033627, 1.36074437, 1.36401015, 1.36878419, 1.37658626, 1.38401983, 1.39675281, 1.40443495, 1.41412555, 1.42713550, 1.43669731, 1.44330612, 1.44907934, 1.44922425, 1.45284731, 1.46112854, 1.46726528, 1.47474834, 1.48625137, 1.50468089, 1.52303800, 1.54314210, 1.55409841, 1.56559874, 1.57796697, 1.58775036, 1.59124341, 1.59983613, 1.61295478, 1.62924563, 1.64488638, 1.66577644, 1.68076843, 1.68799573, 1.69829251, 1.71153919, 1.71752957, 1.72646073, 1.73405716, 1.73544440, 1.73995656, 1.74308848]

cpi = dict(zip(dates, factors))

def data_transf(data0):
    '''
    Transformation of database records

    Data encoding:
    exposure -> real, in years
    pol_type=0 -> valor de mercado referenciado
    pol_type=1 -> valor definido
    veh_age -> integer, beginning in zero
    veh_type=0 -> passeio nacional
    veh_type=1 -> passeio importado
    veh_type=2 -> pick-up's leves nacionais - exceto Kombi e Saveiro
    veh_type=3 -> pick-up's leves nacionais - somente Kombi
    veh_type=4 -> pick-up's leves nacionais - somente Saveiro
    veh_type=5 -> pick-up's leves importados
    veh_type=6 -> modelos esportivos nacionais
    veh_type=7 -> modelos esportivos importados
    veh_type=8 -> modelos especiais (passeio) nacionais
    veh_type=9 -> modelos especiais (passeio) importados
    veh_type=10 -> pick-up's pesadas carga nacionais
    veh_type=11 -> pick-up's pesadas carga importados
    veh_type=12 -> pick-up's pesadas pessoas nacionais
    veh_type=13 -> pick-up's pesadas pessoas importados
    region=0 -> RS - Met. Porto Alegre e Caxias do Sul
    region=1 -> RS - Demais regiões
    region=2 -> SC - Met. Florianópolis e Sul
    region=3 -> SC - Oeste
    region=4 -> SC - Blumenau e demais regiões
    region=5 -> PR - F.Iguaþu-Medianeira-Cascavel-Toledo
    region=6 -> PR - Met. Curitiba
    region=7 -> PR - Demais regiões
    region=8 -> SP - Vale do Paraíba e Ribeira
    region=9 -> SP - Litoral Norte e Baixada Santista
    region=10 -> SP - Met. de São Paulo
    region=11 -> SP - Grande Campinas
    region=12 -> SP - Ribeirão Preto e Demais Mun. de Campinas
    region=13 -> MG - Triângulo mineiro
    region=14 -> MG - Sul
    region=15 -> MG - Met.BH-Centro Oeste-Zona Mata-C. Vertentes
    region=16 -> MG - Vale do Aço-Norte-Vale Jequitinhonha
    region=17 -> RJ - Met. do Rio de Janeiro
    region=18 -> RJ - Interior
    region=19 -> ES - Espírito Santo
    region=20 -> BA - Bahia
    region=21 -> SE - Sergipe
    region=22 -> PE - Pernambuco
    region=23 -> PB - Paraíba
    region=24 -> RN - Rio Grande do Norte
    region=25 -> AL - Alagoas
    region=26 -> CE - Ceará
    region=27 -> PI - Piaui
    region=28 -> MA - Maranhão
    region=29 -> PA - Pará
    region=30 -> AM - Amazonas
    region=31 -> AP - Amapá
    region=32 -> RO - Rondônia
    region=33 -> RR - Roraima
    region=34 -> AC - Acre
    region=35 -> MT - Mato Grosso
    region=36 -> MS - Mato Grosso do Sul
    region=37 -> DF - Brasília
    region=38 -> GO - Goiás
    region=39 -> TO - Tocantins
    region=40 -> GO - Sudeste de Goiás
    sex=0 -> male
    sex=1 -> female
    age -> real, in years/100
    bonus_c=0 -> bonus class '0' (sem bônus)
    bonus_c=1 -> bonus class '1'
    bonus_c=2 -> bonus class '2'
    bonus_c=3 -> bonus class '3'
    bonus_c=4 -> bonus class '4'
    bonus_c=5 -> bonus class '5'
    bonus_c=6 -> bonus class '6'
    bonus_c=7 -> bonus class '7'
    bonus_c=8 -> bonus class '8'
    bonus_c=9 -> bonus class '9'
    bonus_d -> integer, bonus discount in %
    deduct_type=0 -> franquia reduzida
    deduct_type=1 -> franquia normal
    deduct_type=2 -> franquia majorada
    deduct_type=3 -> franquia dedutível
    deduct_type=4 -> sem franquia
    deduct -> real, deductible, valor da franquia 
    cov_casco -> real, coverage vehicle, cobertura casco
    cov_rcdmat -> real, coverage liability, cobertura terceiros danos materiais
    cov_rcdc -> real, coverage liability, cobertura terceiros danos corporais
    cov_rcdmor -> real, coverage liability, cobertura terceiros danos morais
    cov_app_ma -> real, coverage injury, cobertura acidente pessoal morte
    cov_app_ipa -> real, coverage injury, cobertura acidente pessoal invalidez
    cov_app_dmh -> real, coverage injury, cobertura acidente pessoal hospitalar
    pre_casco -> real, premium vehicle, prêmio casco
    pre_rcdmat -> real, premium liability, prêmio terceiros danos materiais
    pre_rcdc -> real, premium liability, prêmio terceiros danos corporais
    pre_rcdmor -> real, premium liability, prêmio terceiros danos morais
    pre_app_ma -> real, premium injury, prêmio acidente pessoal morte
    pre_app_ipa -> real, premium injury, prêmio acidente pessoal invalidez
    pre_app_dmh -> real, premium injury, prêmio acidente pessoal hospitalar
    pre_outros -> real, premium other, prêmio outros
    freq_casco -> integer, claim frequency vehicle
    sev_casco -> real, claim severity vehicle (total)
    sev_cpi_casco -> real, claim severity vehicle (total) cpi adjusted
    freq_rcd -> integer, claim frequency all liabilities
    sev_rcd -> real, claim severity liability (total)
    freq_rcdmat -> integer, claim frequency liability
    sev_rcdmat -> real, claim severity liability (total)
    sev_cpi_rcdmat -> real, claim severity liability (total) cpi adjusted
    freq_rcdc -> integer, claim frequency liability
    sev_rcdc -> real, claim severity liability (total)
    sev_cpi_rcdc -> real, claim severity liability (total) cpi adjusted
    freq_rcdmor -> integer, claim frequency liability
    sev_rcdmor -> real, claim severity liability (total)
    sev_cpi_rcdmor -> real, claim severity liability (total) cpi adjusted
    freq_app -> integer, claim frequency all injuries
    sev_app -> real, claim severity injuty (total)
    freq_app_ma -> integer, claim frequency injury
    sev_app_ma -> real, claim severity injuty (total)
    sev_cpi_app_ma -> real, claim severity injuty (total) cpi adjusted
    freq_app_ipa -> integer, claim frequency injury
    sev_app_ipa -> real, claim severity injuty (total)
    sev_cpi_app_ipa -> real, claim severity injuty (total) cpi adjusted
    freq_app_dmh -> integer, claim frequency injury
    sev_app_dmh -> real, claim severity injuty (total)
    sev_cpi_app_dmh -> real, claim severity injuty (total) cpi adjusted
    freq_outros -> integer, claim frequency other
    sev_outros -> real, claim severity others (total)
    sev_cpi_outros -> real, claim severity others (total) cpi adjusted
    '''

    data = dict(exposure=np.empty(len(data0)), pol_type=np.empty(len(data0), dtype=int), veh_age=np.empty(len(data0), dtype=int), veh_type=np.empty(len(data0), dtype=int), region=np.empty(len(data0), dtype=int), sex=np.empty(len(data0), dtype=int), age=np.empty(len(data0)), bonus_c=np.empty(len(data0), dtype=int), bonus_d=np.empty(len(data0), dtype=int), deduct_type=np.empty(len(data0), dtype=int), deduct=np.empty(len(data0)), cov_casco=np.empty(len(data0)), cov_rcd=np.empty(len(data0)), cov_rcdmat=np.empty(len(data0)), cov_rcdc=np.empty(len(data0)), cov_rcdmor=np.empty(len(data0)), cov_app=np.empty(len(data0)), cov_app_ma=np.empty(len(data0)), cov_app_ipa=np.empty(len(data0)), cov_app_dmh=np.empty(len(data0)), pre_casco=np.empty(len(data0)), pre_rcdmat=np.empty(len(data0)), pre_rcdc=np.empty(len(data0)), pre_rcdmor=np.empty(len(data0)), pre_app_ma=np.empty(len(data0)), pre_app_ipa=np.empty(len(data0)), pre_app_dmh=np.empty(len(data0)), pre_outros=np.empty(len(data0)),freq_casco=np.zeros(len(data0), dtype=int), sev_casco=np.zeros(len(data0)), freq_rcd=np.zeros(len(data0), dtype=int), sev_rcd=np.zeros(len(data0)), freq_rcdmat=np.zeros(len(data0), dtype=int), sev_rcdmat=np.zeros(len(data0)), freq_rcdc=np.zeros(len(data0), dtype=int), sev_rcdc=np.zeros(len(data0)), freq_rcdmor=np.zeros(len(data0), dtype=int), sev_rcdmor=np.zeros(len(data0)), freq_app=np.zeros(len(data0), dtype=int), sev_app=np.zeros(len(data0)), freq_app_ma=np.zeros(len(data0), dtype=int), sev_app_ma=np.zeros(len(data0)), freq_app_ipa=np.zeros(len(data0), dtype=int), sev_app_ipa=np.zeros(len(data0)), freq_app_dmh=np.zeros(len(data0), dtype=int), sev_app_dmh=np.zeros(len(data0)), freq_outros=np.zeros(len(data0), dtype=int), sev_outros=np.zeros(len(data0)))

    for k, x in enumerate(data0):
        if x[0] == '1':
            data['pol_type'][k] = 0
        elif x[0] == '2':
            data['pol_type'][k] = 1
        
        data['veh_age'][k] = max(0, int(str(x[4])[:4]) - x[1])

        if x[2].strip() == '10':
            data['veh_type'][k] = 0
        elif x[2].strip() == '11':
            data['veh_type'][k] = 1
        elif x[2] == '14A':
            data['veh_type'][k] = 2
        elif x[2] == '14B':
            data['veh_type'][k] = 3
        elif x[2] == '14C':
            data['veh_type'][k] = 4
        elif x[2].strip() == '15':
            data['veh_type'][k] = 5
        elif x[2].strip() == '16':
            data['veh_type'][k] = 6
        elif x[2].strip() == '17':
            data['veh_type'][k] = 7
        elif x[2].strip() == '18':
            data['veh_type'][k] = 8
        elif x[2].strip() == '19':
            data['veh_type'][k] = 9
        elif x[2].strip() == '20':
            data['veh_type'][k] = 10
        elif x[2].strip() == '21':
            data['veh_type'][k] = 11
        elif x[2].strip() == '22':
            data['veh_type'][k] = 12
        elif x[2].strip() == '23':
            data['veh_type'][k] = 13

        if x[3] == '01':
            data['region'][k] = 0
        elif x[3] == '02':
            data['region'][k] = 1
        elif x[3] == '03':
            data['region'][k] = 2
        elif x[3] == '04':
            data['region'][k] = 3
        elif x[3] == '05':
            data['region'][k] = 4
        elif x[3] == '06':
            data['region'][k] = 5
        elif x[3] == '07':
            data['region'][k] = 6
        elif x[3] == '08':
            data['region'][k] = 7
        elif x[3] == '09':
            data['region'][k] = 8
        elif x[3] == '10':
            data['region'][k] = 9
        elif x[3] == '11':
            data['region'][k] = 10
        elif x[3] == '12':
            data['region'][k] = 11
        elif x[3] == '13':
            data['region'][k] = 12
        elif x[3] == '14':
            data['region'][k] = 13
        elif x[3] == '15':
            data['region'][k] = 14
        elif x[3] == '16':
            data['region'][k] = 15
        elif x[3] == '17':
            data['region'][k] = 16
        elif x[3] == '18':
            data['region'][k] = 17
        elif x[3] == '19':
            data['region'][k] = 18
        elif x[3] == '20':
            data['region'][k] = 19
        elif x[3] == '21':
            data['region'][k] = 20
        elif x[3] == '22':
            data['region'][k] = 21
        elif x[3] == '23':
            data['region'][k] = 22
        elif x[3] == '24':
            data['region'][k] = 23
        elif x[3] == '25':
            data['region'][k] = 24
        elif x[3] == '26':
            data['region'][k] = 25
        elif x[3] == '27':
            data['region'][k] = 26
        elif x[3] == '28':
            data['region'][k] = 27
        elif x[3] == '29':
            data['region'][k] = 28
        elif x[3] == '30':
            data['region'][k] = 29
        elif x[3] == '31':
            data['region'][k] = 30
        elif x[3] == '32':
            data['region'][k] = 31
        elif x[3] == '33':
            data['region'][k] = 32
        elif x[3] == '34':
            data['region'][k] = 33
        elif x[3] == '35':
            data['region'][k] = 34
        elif x[3] == '36':
            data['region'][k] = 35
        elif x[3] == '37':
            data['region'][k] = 36
        elif x[3] == '38':
            data['region'][k] = 37
        elif x[3] == '39':
            data['region'][k] = 38
        elif x[3] == '40':
            data['region'][k] = 39
        elif x[3] == '41':
            data['region'][k] = 40

        if x[6] == 'M':
            data['sex'][k] = 0
        elif x[6] == 'F':
            data['sex'][k] = 1

        aux_age = relativedelta(x[4], x[7]).years
        aux_age += relativedelta(x[4], x[7]).months / 12
        aux_age += relativedelta(x[4], x[7]).days / 365.2425
        data['age'][k] = aux_age / 100

        if x[10] == '0':
            data['bonus_c'][k] = 0
        elif x[10] == '1':
            data['bonus_c'][k] = 1
        elif x[10] == '2':
            data['bonus_c'][k] = 2
        elif x[10] == '3':
            data['bonus_c'][k] = 3
        elif x[10] == '4':
            data['bonus_c'][k] = 4
        elif x[10] == '5':
            data['bonus_c'][k] = 5
        elif x[10] == '6':
            data['bonus_c'][k] = 6
        elif x[10] == '7':
            data['bonus_c'][k] = 7
        elif x[10] == '8':
            data['bonus_c'][k] = 8
        elif x[10] == '9':
            data['bonus_c'][k] = 9

        data['bonus_d'][k] = x[28]

        if x[11] == '1':
            data['deduct_type'][k] = 0
        elif x[11] == '2':
            data['deduct_type'][k] = 1
        elif x[11] == '3':
            data['deduct_type'][k] = 2
        elif x[11] == '4':
            data['deduct_type'][k] = 3
        elif x[11] == '9':
            data['deduct_type'][k] = 4

        data['deduct'][k] = (x[12] / cpi[str(x[4])[:-3]]) / 1000

        data['cov_casco'][k] = (x[13] / cpi[str(x[4])[:-3]]) / 1000
        data['cov_rcdmat'][k] = (x[14] / cpi[str(x[4])[:-3]]) / 1000
        data['cov_rcdc'][k] = (x[15] / cpi[str(x[4])[:-3]]) / 1000
        data['cov_rcdmor'][k] = (x[16] / cpi[str(x[4])[:-3]]) / 1000
        data['cov_rcd'][k] = ((x[14] + x[15] + x[16]) / cpi[str(x[4])[:-3]]) / 1000
        data['cov_app_ma'][k] = (x[17] / cpi[str(x[4])[:-3]]) / 1000
        data['cov_app_ipa'][k] = (x[18] / cpi[str(x[4])[:-3]]) / 1000
        data['cov_app_dmh'][k] = (x[19] / cpi[str(x[4])[:-3]]) / 1000
        data['cov_app'][k] = ((x[17] + x[18] + x[19]) / cpi[str(x[4])[:-3]]) / 1000
        data['pre_casco'][k] = (x[20] / cpi[str(x[4])[:-3]]) / 1000
        data['pre_rcdmat'][k] = (x[21] / cpi[str(x[4])[:-3]]) / 1000
        data['pre_rcdc'][k] = (x[22] / cpi[str(x[4])[:-3]]) / 1000
        data['pre_rcdmor'][k] = (x[23] / cpi[str(x[4])[:-3]]) / 1000
        data['pre_app_ma'][k] = (x[24] / cpi[str(x[4])[:-3]]) / 1000
        data['pre_app_ipa'][k] = (x[25] / cpi[str(x[4])[:-3]]) / 1000
        data['pre_app_dmh'][k] = (x[26] / cpi[str(x[4])[:-3]]) / 1000
        data['pre_outros'][k] = (x[27] / cpi[str(x[4])[:-3]]) / 1000

        if x[8] != None:
            aux8 = list(zip(re.findall(r' "(\d)" ', x[8]), re.findall(r'"f1":(\d+)', x[8]), re.findall(r'"f2":"(\d+-\d+-\d+)"', x[8])))
        if x[9] != None:
            aux9 = list(zip(re.findall(r'"f1":"(\d)"', x[9]), re.findall(r'"f2":"(\d+-\d+-\d+)"', x[9])))

        if x[8] == None and x[9] == None:
            delta_years = relativedelta(x[5], x[4]).years
            delta_years += relativedelta(x[5], x[4]).months / 12
            delta_years += relativedelta(x[5], x[4]).days / 365.2425
            data['exposure'][k] = delta_years
        elif x[8] == None and x[9] != None:
            fim_vig = x[5]
            for i in aux9:
                if i[0] in {'1', '2', '3'}:
                    if (datetime.strptime(i[1], '%Y-%m-%d').date()-x[4]).days <= 0:
                        fim_vig = x[4]
                    else:
                        if (datetime.strptime(i[1], '%Y-%m-%d').date()-fim_vig).days < 0:
                            fim_vig = datetime.strptime(i[1], '%Y-%m-%d').date()

            delta_years = relativedelta(fim_vig, x[4]).years
            delta_years += relativedelta(fim_vig, x[4]).months / 12
            delta_years += relativedelta(fim_vig, x[4]).days / 365.2425
            data['exposure'][k] = delta_years
        elif x[8] != None and x[9] == None:
            delta_years = relativedelta(x[5], x[4]).years
            delta_years += relativedelta(x[5], x[4]).months / 12
            delta_years += relativedelta(x[5], x[4]).days / 365.2425
            data['exposure'][k] = delta_years

            aux_dict_casco = {}
            aux_dict_rcd = {}
            aux_dict_rcdmat = {}
            aux_dict_rcdc = {}
            aux_dict_rcdmor = {}
            aux_dict_app = {}
            aux_dict_app_ma = {}
            aux_dict_app_ipa = {}
            aux_dict_app_dmh = {}
            aux_dict_outros = {}
            for i in aux8:
                # remove claims where year is bellow inicio_vig:
                if int(i[2][:4]) < int(str(x[4])[:4]):
                    continue

                if i[0] in {'1'}:
                    if float(i[1]) >= 10:
                        if i[2] not in aux_dict_casco.keys():
                            aux_dict_casco[i[2]] = float(i[1]) / cpi[i[2][:-3]]
                        else:
                            aux_dict_casco[i[2]] += float(i[1]) / cpi[i[2][:-3]]
                elif i[0] in {'2', '3', '4'}:
                    if float(i[1]) >= 10:
                        if i[2] not in aux_dict_rcd.keys():
                            aux_dict_rcd[i[2]] = float(i[1]) / cpi[i[2][:-3]]
                        else:
                            aux_dict_rcd[i[2]] += float(i[1]) / cpi[i[2][:-3]]
                        if i[0] == '2':
                            if i[2] not in aux_dict_rcdmat.keys():
                                aux_dict_rcdmat[i[2]] = float(i[1]) / cpi[i[2][:-3]]
                            else:
                                aux_dict_rcdmat[i[2]] += float(i[1]) / cpi[i[2][:-3]]
                        if i[0] == '3':
                            if i[2] not in aux_dict_rcdc.keys():
                                aux_dict_rcdc[i[2]] = float(i[1]) / cpi[i[2][:-3]]
                            else:
                                aux_dict_rcdc[i[2]] += float(i[1]) / cpi[i[2][:-3]]
                        if i[0] == '4':
                            if i[2] not in aux_dict_rcdmor.keys():
                                aux_dict_rcdmor[i[2]] = float(i[1]) / cpi[i[2][:-3]]
                            else:
                                aux_dict_rcdmor[i[2]] += float(i[1]) / cpi[i[2][:-3]]
                elif i[0] in {'5', '6', '7'}:
                    if float(i[1]) >= 10:
                        if i[2] not in aux_dict_app.keys():
                            aux_dict_app[i[2]] = float(i[1]) / cpi[i[2][:-3]]
                        else:
                            aux_dict_app[i[2]] += float(i[1]) / cpi[i[2][:-3]]
                        if i[0] == '5':
                            if i[2] not in aux_dict_app_ma.keys():
                                aux_dict_app_ma[i[2]] = float(i[1]) / cpi[i[2][:-3]]
                            else:
                                aux_dict_app_ma[i[2]] += float(i[1]) / cpi[i[2][:-3]]
                        if i[0] == '6':
                            if i[2] not in aux_dict_app_ipa.keys():
                                aux_dict_app_ipa[i[2]] = float(i[1]) / cpi[i[2][:-3]]
                            else:
                                aux_dict_app_ipa[i[2]] += float(i[1]) / cpi[i[2][:-3]]
                        if i[0] == '7':
                            if i[2] not in aux_dict_app_dmh.keys():
                                aux_dict_app_dmh[i[2]] = float(i[1]) / cpi[i[2][:-3]]
                            else:
                                aux_dict_app_dmh[i[2]] += float(i[1]) / cpi[i[2][:-3]]
                elif i[0] in {'8'}:
                    if float(i[1]) >= 5:
                        if i[2] not in aux_dict_outros.keys():
                            aux_dict_outros[i[2]] = float(i[1]) / cpi[i[2][:-3]]
                        else:
                            aux_dict_outros[i[2]] += float(i[1]) / cpi[i[2][:-3]]

            data['freq_casco'][k] = len(aux_dict_casco.values())
            data['sev_casco'][k] = sum(aux_dict_casco.values()) / 1000
            data['freq_rcd'][k] = len(aux_dict_rcd.values())
            data['sev_rcd'][k] = sum(aux_dict_rcd.values()) / 1000
            data['freq_rcdmat'][k] = len(aux_dict_rcdmat.values())
            data['sev_rcdmat'][k] = sum(aux_dict_rcdmat.values()) / 1000
            data['freq_rcdc'][k] = len(aux_dict_rcdc.values())
            data['sev_rcdc'][k] = sum(aux_dict_rcdc.values()) / 1000
            data['freq_rcdmor'][k] = len(aux_dict_rcdmor.values())
            data['sev_rcdmor'][k] = sum(aux_dict_rcdmor.values()) / 1000
            data['freq_app'][k] = len(aux_dict_app.values())
            data['sev_app'][k] = sum(aux_dict_app.values()) / 1000
            data['freq_app_ma'][k] = len(aux_dict_app_ma.values())
            data['sev_app_ma'][k] = sum(aux_dict_app_ma.values()) / 1000
            data['freq_app_ipa'][k] = len(aux_dict_app_ipa.values())
            data['sev_app_ipa'][k] = sum(aux_dict_app_ipa.values()) / 1000
            data['freq_app_dmh'][k] = len(aux_dict_app_dmh.values())
            data['sev_app_dmh'][k] = sum(aux_dict_app_dmh.values()) / 1000
            data['freq_outros'][k] = len(aux_dict_outros.values())
            data['sev_outros'][k] = sum(aux_dict_outros.values()) / 1000
        else:
            fim_vig = x[5]
            for i in aux9:
                if i[0] in {'1', '2', '3'}:
                    if (datetime.strptime(i[1], '%Y-%m-%d').date()-x[4]).days <= 0:
                        fim_vig = x[4]
                    else:
                        if (datetime.strptime(i[1], '%Y-%m-%d').date()-fim_vig).days < 0:
                            fim_vig = datetime.strptime(i[1], '%Y-%m-%d').date()
            
            delta_years = relativedelta(fim_vig, x[4]).years
            delta_years += relativedelta(fim_vig, x[4]).months / 12
            delta_years += relativedelta(fim_vig, x[4]).days / 365.2425
            data['exposure'][k] = delta_years

            aux_dict_casco = {}
            aux_dict_rcd = {}
            aux_dict_rcdmat = {}
            aux_dict_rcdc = {}
            aux_dict_rcdmor = {}
            aux_dict_app = {}
            aux_dict_app_ma = {}
            aux_dict_app_ipa = {}
            aux_dict_app_dmh = {}
            aux_dict_outros = {}
            for i in aux8:
                # remove claims where year is bellow inicio_vig:
                if int(i[2][:4]) < int(str(x[4])[:4]):
                    continue

                if i[0] in {'1'}:
                    if float(i[1]) >= 10 and (fim_vig-datetime.strptime(i[2], '%Y-%m-%d').date()).days >= 0:
                        if i[2] not in aux_dict_casco.keys():
                            aux_dict_casco[i[2]] = float(i[1]) / cpi[i[2][:-3]]
                        else:
                            aux_dict_casco[i[2]] += float(i[1]) / cpi[i[2][:-3]]
                elif i[0] in {'2', '3', '4'}:
                    if float(i[1]) >= 10 and (fim_vig-datetime.strptime(i[2], '%Y-%m-%d').date()).days >= 0:
                        if i[2] not in aux_dict_rcd.keys():
                            aux_dict_rcd[i[2]] = float(i[1]) / cpi[i[2][:-3]]
                        else:
                            aux_dict_rcd[i[2]] += float(i[1]) / cpi[i[2][:-3]]
                        if i[0] == '2':
                            if i[2] not in aux_dict_rcdmat.keys():
                                aux_dict_rcdmat[i[2]] = float(i[1]) / cpi[i[2][:-3]]
                            else:
                                aux_dict_rcdmat[i[2]] += float(i[1]) / cpi[i[2][:-3]]
                        if i[0] == '3':
                            if i[2] not in aux_dict_rcdc.keys():
                                aux_dict_rcdc[i[2]] = float(i[1]) / cpi[i[2][:-3]]
                            else:
                                aux_dict_rcdc[i[2]] += float(i[1]) / cpi[i[2][:-3]]
                        if i[0] == '4':
                            if i[2] not in aux_dict_rcdmor.keys():
                                aux_dict_rcdmor[i[2]] = float(i[1]) / cpi[i[2][:-3]]
                            else:
                                aux_dict_rcdmor[i[2]] += float(i[1]) / cpi[i[2][:-3]]
                elif i[0] in {'5', '6', '7'} and (fim_vig-datetime.strptime(i[2], '%Y-%m-%d').date()).days >= 0:
                    if float(i[1]) >= 10:
                        if i[2] not in aux_dict_app.keys():
                            aux_dict_app[i[2]] = float(i[1]) / cpi[i[2][:-3]]
                        else:
                            aux_dict_app[i[2]] += float(i[1]) / cpi[i[2][:-3]]
                        if i[0] == '5':
                            if i[2] not in aux_dict_app_ma.keys():
                                aux_dict_app_ma[i[2]] = float(i[1]) / cpi[i[2][:-3]]
                            else:
                                aux_dict_app_ma[i[2]] += float(i[1]) / cpi[i[2][:-3]]
                        if i[0] == '6':
                            if i[2] not in aux_dict_app_ipa.keys():
                                aux_dict_app_ipa[i[2]] = float(i[1]) / cpi[i[2][:-3]]
                            else:
                                aux_dict_app_ipa[i[2]] += float(i[1]) / cpi[i[2][:-3]]
                        if i[0] == '7':
                            if i[2] not in aux_dict_app_dmh.keys():
                                aux_dict_app_dmh[i[2]] = float(i[1]) / cpi[i[2][:-3]]
                            else:
                                aux_dict_app_dmh[i[2]] += float(i[1]) / cpi[i[2][:-3]]
                elif i[0] in {'8'} and (fim_vig-datetime.strptime(i[2], '%Y-%m-%d').date()).days >= 0:
                    if float(i[1]) >= 5:
                        if i[2] not in aux_dict_outros.keys():
                            aux_dict_outros[i[2]] = float(i[1]) / cpi[i[2][:-3]]
                        else:
                            aux_dict_outros[i[2]] += float(i[1]) / cpi[i[2][:-3]]

            data['freq_casco'][k] = len(aux_dict_casco.values())
            data['sev_casco'][k] = sum(aux_dict_casco.values()) / 1000
            data['freq_rcd'][k] = len(aux_dict_rcd.values())
            data['sev_rcd'][k] = sum(aux_dict_rcd.values()) / 1000
            data['freq_rcdmat'][k] = len(aux_dict_rcdmat.values())
            data['sev_rcdmat'][k] = sum(aux_dict_rcdmat.values()) / 1000
            data['freq_rcdc'][k] = len(aux_dict_rcdc.values())
            data['sev_rcdc'][k] = sum(aux_dict_rcdc.values()) / 1000
            data['freq_rcdmor'][k] = len(aux_dict_rcdmor.values())
            data['sev_rcdmor'][k] = sum(aux_dict_rcdmor.values()) / 1000
            data['freq_app'][k] = len(aux_dict_app.values())
            data['sev_app'][k] = sum(aux_dict_app.values()) / 1000
            data['freq_app_ma'][k] = len(aux_dict_app_ma.values())
            data['sev_app_ma'][k] = sum(aux_dict_app_ma.values()) / 1000
            data['freq_app_ipa'][k] = len(aux_dict_app_ipa.values())
            data['sev_app_ipa'][k] = sum(aux_dict_app_ipa.values()) / 1000
            data['freq_app_dmh'][k] = len(aux_dict_app_dmh.values())
            data['sev_app_dmh'][k] = sum(aux_dict_app_dmh.values()) / 1000
            data['freq_outros'][k] = len(aux_dict_outros.values())
            data['sev_outros'][k] = sum(aux_dict_outros.values()) / 1000
    return data

def elim_zeroexp(data):
    '''Eliminates obs for which length of exposure is less than a week'''

    index = np.where(data['exposure'] > 0.25/12)[0]
    for key in data.keys():
        data[key] = data[key][index]

    return data

def elim_toomanyclaims(data):
    '''Eliminates obs for which number of claims is greater than 15'''

    for aux in ('casco', 'rcd', 'app'):
        index = np.where(data['freq_' + aux] > 15)[0]
        for key in data.keys():
            data[key] = data[key][index]

    return data

## function elim_specialmods currently not in use
def elim_specialmods(data):
    '''Eliminates obs for which vehicle type is special model (too few obs)'''

    index = np.where(np.logical_and(data['veh_type'] != 8, data['veh_type'] != 9))[0]
    for key in data.keys():
        data[key] = data[key][index]

    return data

def save_results(data, mmm, aa):
    try:
        os.remove(data_dir + 'data_' + mmm + aa + '.pkl')
    except OSError:
        pass

    with open(data_dir + 'data_' + mmm + aa + '.pkl', 'wb') as filename:
        pickle.dump(data, filename)

    print('Data saved for period ' + mmm + aa) 
    return


sql_code = '''SELECT cod_cont, ano_modelo, cod_tarif, regiao, inicio_vig, fim_vig, sexo, data_nasc, sinistro::text, endosso::text, clas_bonus, tipo_franq, val_franq, is_casco, is_rcdmat, is_rcdc, is_rcdmor, is_app_ma, is_app_ipa, is_app_dmh, pre_casco, pre_rcdmat, pre_rcdc, pre_rcdmor, pre_app_ma, pre_app_ia, pre_app_dm, pre_outros, perc_bonus FROM rs_:mmm:aa WHERE tipo_pes = 'F' AND cobertura = '1' AND cod_cont in ('1', '2') AND ano_modelo > 1900 AND ano_modelo < 2018 AND cod_tarif in (' 10', '10 ', ' 11', '11 ', '14A', '14B', '14C', ' 15', '15 ', ' 16', '16 ', ' 17', '17 ', ' 18', '18 ', ' 19', '19 ', ' 20', '20 ', ' 21', '21 ', ' 22', '22 ', ' 23', '23 ') AND regiao in ('01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', '35', '36', '37', '38', '39', '40', '41') AND sexo in ('F', 'M') AND data_nasc > '1900-01-01' AND date_part('year', data_nasc) < date_part('year', inicio_vig) - 16 AND clas_bonus in ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9') AND tipo_franq in ('1', '2', '3', '4', '9') AND val_franq >= 0 AND val_franq < 1e6 AND is_casco >= 0 AND is_casco < 1e7 AND is_rcdmat >= 0 AND is_rcdmat < 5e7 AND is_rcdc >= 0 AND is_rcdc < 5e7 AND is_rcdmor >= 0 AND is_rcdmor < 5e7 AND is_app_ma >= 0 AND is_app_ma < 5e7 AND is_app_ipa >= 0 AND is_app_ipa < 5e7 AND is_app_dmh >= 0 AND is_app_dmh < 5e7 AND pre_casco >= 0 AND pre_casco < 1e6 AND pre_rcdmat >= 0 AND pre_rcdmat < 1e6 AND pre_rcdc >= 0 AND pre_rcdc < 1e6 AND pre_rcdmor >= 0 AND pre_rcdmor < 1e6 AND pre_app_ma >= 0 AND pre_app_ma < 1e6 AND pre_app_ia >= 0 AND pre_app_ia < 1e6 AND pre_app_dm >= 0 AND pre_app_dm < 1e6 AND pre_outros >= 0 AND pre_outros < 1e6 AND perc_bonus >= 0 AND perc_bonus < 99;'''

if __name__ == '__main__':
    years = ('08', '09', '10', '11')
    months = ('jan', 'fev', 'mar', 'abr', 'mai', 'jun', 'jul', 'ago', 'set', 'out', 'nov', 'dez')
    for aa in years:
        for mmm in months:
            sql_code2 = sql_code.replace(':aa', aa)
            sql_code2 = sql_code2.replace(':mmm', mmm)
            conn = psycopg2.connect("dbname=susep user=ricardob")
            cur = conn.cursor()
            cur.execute(sql_code2)
            data0 = cur.fetchall()
            conn.commit()
            cur.close()
            conn.close()
            data = data_transf(data0)
            data = elim_zeroexp(data)
            data = elim_toomanyclaims(data) 
            save_results(data, mmm, aa)
