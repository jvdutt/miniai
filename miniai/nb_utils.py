# cell starts
import json      
# cell ends

# cell starts
def exportnb(nb_path,f_path,mode = 'w'):
    
    with open(nb_path, mode= "r", encoding= "utf-8") as f:
        nb = json.load(f)
        
    new_doc=[]
    for cell in nb['cells']:
        if len(cell['source'])>1 and ('#/export' in cell['source'][0]):new_doc.append(cell['source'][1:])
    
    with open(f_path,mode) as f:
        for cell in new_doc:
            f.write('# cell starts\n')
            f.writelines(cell);f.write('\n')
            f.write('# cell ends\n\n') 
            
    
# cell ends

