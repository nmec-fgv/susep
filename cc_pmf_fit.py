#############################################################################
## Calcs parameter estimates for mixed Poisson pmf fit of claim count data ##
#############################################################################

filename = 'claim_counts.pkl'
try:
    os.path.exists('/home/ricardob/Susep/Data/' + filename)
    with open('/home/ricardob/Susep/Data/' + filename, 'rb') as file:
        data = pickle.load(file)
except:
    print('File ' + filename + ' not found')
return data
