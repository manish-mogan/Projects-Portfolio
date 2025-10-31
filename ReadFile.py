# File: ReadFile.py
# Description: Reads the Yale Bright Star Catalog (bsc5.dat) and converts it to CSV format
# Name: Manish Mogan and Ritesh Penumatsa
# UT EID: mm86873 and rp37458
# Date: October 26, 2025

import sys
import csv

def parse_line(line):
    if len(line) < 197:
        line = line.ljust(197)
    
    hr = line[0:4].strip()
    name = line[4:14].strip()
    dmid = line[14:25].strip()
    hd = line[25:31].strip()
    sao = line[31:37].strip()
    fk5 = line[37:41].strip()
    irflag = line[41].strip()
    multiple = line[43].strip()
    ads = line[44:49].strip()
    varname = line[51:60].strip()
    
    rah = line[75:77].strip()
    ram = line[77:79].strip()
    ras = line[79:83].strip()
    
    decsign = line[83].strip()
    decd = line[84:86].strip()
    decm = line[86:88].strip()
    decs = line[88:90].strip()
    
    glon = line[90:96].strip()
    glat = line[96:102].strip()
    
    vmag = line[102:107].strip()
    bv = line[109:114].strip()
    ub = line[115:120].strip()
    ri = line[121:126].strip()
    
    sptype = line[127:147].strip()
    
    pmra = line[148:154].strip()
    pmdec = line[154:160].strip()
    
    parallax = line[161:166].strip()
    radvel = line[166:170].strip()
    
    rotatvel = line[176:179].strip()
    dmag1 = line[180:184].strip()
    dmag2 = line[184:189].strip()
    orbit = line[189:190].strip()
    notes = line[190:196].strip()
    
    return {
        'HR': hr,
        'Name': name,
        'DM': dmid,
        'HD': hd,
        'SAO': sao,
        'FK5': fk5,
        'IRFlag': irflag,
        'Multiple': multiple,
        'ADS': ads,
        'VarName': varname,
        'RAh': rah,
        'RAm': ram,
        'RAs': ras,
        'DecSign': decsign,
        'DecD': decd,
        'DecM': decm,
        'DecS': decs,
        'GLON': glon,
        'GLAT': glat,
        'Vmag': vmag,
        'BV': bv,
        'UB': ub,
        'RI': ri,
        'SpType': sptype,
        'PMRA': pmra,
        'PMDec': pmdec,
        'Parallax': parallax,
        'RadVel': radvel,
        'RotVel': rotatvel,
        'DMag1': dmag1,
        'DMag2': dmag2,
        'Orbit': orbit,
        'Notes': notes
    }

def main():
    input_file = 'bsc5.dat'
    output_file = 'bsc5.csv'
    
    fieldnames = [
        'HR', 'Name', 'DM', 'HD', 'SAO', 'FK5', 'IRFlag', 'Multiple', 'ADS', 'VarName',
        'RAh', 'RAm', 'RAs', 'DecSign', 'DecD', 'DecM', 'DecS',
        'GLON', 'GLAT', 'Vmag', 'BV', 'UB', 'RI', 'SpType',
        'PMRA', 'PMDec', 'Parallax', 'RadVel', 'RotVel', 'DMag1', 'DMag2', 'Orbit', 'Notes'
    ]
    
    try:
        with open(input_file, 'r', encoding='latin-1') as infile:
            with open(output_file, 'w', newline='', encoding='utf-8') as outfile:
                writer = csv.DictWriter(outfile, fieldnames=fieldnames)
                writer.writeheader()
                
                for line in infile:
                    if line.strip():
                        data = parse_line(line)
                        writer.writerow(data)
        
        print(f"Successfully converted {input_file} to {output_file}")
        
    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
