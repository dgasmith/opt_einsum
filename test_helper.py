import numpy as np

### Build dictionary of tests
tests = {}
# Randomly produced contractions
tests['Random1'] = ['aab,fa,df,ecc->bde',  [25, 18, 23, 16, 28, 14]]
tests['Random2'] = ['ecb,fef,bad,ed->ac',  [19, 20, 17, 21, 12, 21]]

# Index transformations
tests['Index1'] = ['ea,fb,abcd,gc,hd->efgh', [10, 10, 10, 10, 15, 8, 6, 3]]
tests['Index2'] = ['ea,fb,abcd,gc,hd->efgh', [10, 10, 10, 10, 3, 6, 8, 15]]
tests['Index3'] = ['ea,fb,abcd,gc,hd->efgh', [15, 8, 6, 3, 10, 10, 10, 10]]

# Hadamard like
tests['Hadamard1'] = ['abc,abc->abc', [200, 200, 200]]
tests['Hadamard2'] = ['abc,abc,abc->abc', [200, 200, 200]]
tests['Hadamard3'] = ['abc,ab,abc->abc', [200, 200, 200]]
tests['Hadamard4'] = ['a,ab,abc->abc', [200, 200, 200]]

# Real world test cases
tests['EP_Theory1'] = ['acjl,pbpk,jkib,ilac,jlac,jklabc,ilac', [10, 5, 9, 10, 5, 25, 6, 14, 11]]
tests['EP_Theory2'] = ['cj,bdik,akdb,ijca,jc,ijkbcd,ijac', [20, 14, 9, 10, 9, 12, 13, 14, 11]]
tests['EP_Theory3'] = ['abik,ikjp,pjba,ikab,jab', [10, 22, 15, 26, 17, 25]]
tests['EP_Theory4'] = ['bdk,cji,ajdb,ikca,kbd,ijkcd,ikac', [10, 11, 9, 10, 12, 15, 13, 14, 11]]
tests['EP_Theory5'] = ['cij,bdk,ajbc,ikad,ijc,ijk,ikad', [10, 17, 9, 10, 13, 16, 15, 14, 11]]
tests['EP_Theory6'] = ['cij,bdk,ajbc,ikad,bdk,cji,ajdb', [10, 17, 9, 10, 13, 16, 15, 14, 11]]
#tests['Actual2'] = [, [10, 5, 9, 10, 5, 25, 6, 14, 11]]

# A few tricky cases
tests['Collapse1'] = ['ab,ab,c->', [400, 400, 400]]
tests['Collapse2'] = ['ab,ab,c->c', [400, 400, 400]]
tests['Collapse3'] = ['ab,ab,cd,cd->', [60, 60, 60, 60]]
tests['Collapse4'] = ['ab,ab,cd,cd->ac', [60, 60, 60, 60]]
tests['Collapse5'] = ['ab,ab,cd,cd->cd', [60, 60, 60, 60]]
tests['Collapse6'] = ['ab,ab,cd,cd,ef,ef->', [15, 15, 15, 15, 15, 15]]

tests['Expand1'] = ['ab,cd,ef->abcdef', [15, 14, 13, 12, 11, 10]]
tests['Expand2'] = ['ab,cd,ef->acdf', [15, 14, 13, 12, 11, 10]]
tests['Expand3'] = ['ab,cd,de->abcde', [15, 14, 13, 12, 11, 10]]
tests['Expand4'] = ['ab,cd,de->be', [15, 14, 13, 12, 11, 10]]
tests['Expand5'] = ['ab,bcd,cd->abcd', [40, 39, 37, 36]]
tests['Expand6'] = ['ab,bcd,cd->abd', [40, 39, 37, 36]]

# Random test cases that have previously failed
tests['Failed1'] = ['eb,cb,fb->cef', [60, 59, 48, 57]]
tests['Failed2'] = ['dd,fb,be,cdb->cef', [27, 28, 13, 18, 19, 20]]
tests['Failed3'] = ['bca,cdb,dbf,afc->', [15, 27, 22, 17, 18, 29]]

# GEMM test
tests['Dot1'] = ['ab,bc', [400, 401, 402]]
tests['Dot2'] = ['abc,bc', [400, 401, 402]]
tests['Dot3'] = ['abc,acb->a', [200, 201, 202]]
tests['Dot4'] = ['abcd,cdef', [30, 29, 28, 27, 26, 25]]
tests['Dot5'] = ['abcd,efdc', [19, 18, 17, 16, 15, 14]]
tests['Dot6'] = ['abcd,defc', [19, 18, 17, 16, 15, 14]]

# Previous test showed that opt_einsum is 2-10x slower than einsum
tests['Slow1'] = ['bcf,bbb,fbf,fc->', [15, 25, 10, 10, 12, 13]]
tests['Slow2'] = ['bb,ff,be->e', [27, 24, 27, 25, 17, 22]]
tests['Slow3'] = ['ad,dd,dc->ac', [29, 20, 26, 20, 24, 13]]
tests['Slow4'] = ['bcb,bb,fc,fff->', [12, 22, 18, 22, 19, 29]] 
tests['Slow5'] = ['fc,dcf,fad->a', [13, 15, 18, 14, 27, 22]]
tests['Slow6'] = ['cec,ed,cd,ec->', [18, 22, 15, 17, 22, 19]]

def build_views(string, sizes):
    '''
    string- einsum like string 'abc,cb->ab'
    sizes- 
    '''
    terms = string.split('->')[0].split(',')

    alpha = ''.join(set(''.join(terms)))
    sizes_dict = {alpha:size for alpha,size in zip(alpha, sizes)}

    views = []
    for term in terms:
        term_index = [sizes_dict[x] for x in term]
        views.append(np.random.rand(*term_index))
    return views


alpha = list('abcdefghijklmnopqrstuvwyxz')
alpha_dict = {num:x for num, x in enumerate(alpha)}

