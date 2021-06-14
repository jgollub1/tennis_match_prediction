'''
clean up mispellings in datasets. specific to atp/wta tours
'''
def normalize_name(s, tour='atp'):
    if tour=='atp':
        s = s.replace('-',' ')
        s = s.replace('Stanislas','Stan').replace('Stan','Stanislas')
        s = s.replace('Alexandre','Alexander')
        s = s.replace('Federico Delbonis','Federico Del').replace('Federico Del','Federico Delbonis')
        s = s.replace('Mello','Melo')
        s = s.replace('Cedric','Cedrik')
        s = s.replace('Bernakis','Berankis')
        s = s.replace('Hansescu','Hanescu')
        s = s.replace('Teimuraz','Teymuraz')
        s = s.replace('Vikor','Viktor')
        s = s.rstrip()
        s = s.replace('Alex Jr.','Alex Bogomolov')
        s = s.title()
        sep = s.split(' ')
        return ' '.join(sep[:2]) if len(sep)>2 else s
    else:
        return s