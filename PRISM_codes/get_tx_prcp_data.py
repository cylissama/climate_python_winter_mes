import os
from datetime import datetime, timedelta
from ftplib import FTP

# Configuration
output_dir = '../output_data'
variables = ['tmean', 'ppt']  # TMEAN and PRCP (ppt is precipitation)
base_url = 'prism.oregonstate.edu'
stabilization_period = 6  # Months before data is considered final

# Calculate date ranges with stabilization buffer
current_date = datetime.now()
end_date = (current_date.replace(day=1) - timedelta(days=1))  # Last complete month
cutoff_date = (end_date - timedelta(days=stabilization_period*30)).replace(day=1)
start_date = end_date - timedelta(days=5*365)  # 5 years back

def get_remote_mtime(ftp, filename):
    """Get last modification time of remote file"""
    try:
        resp = ftp.voidcmd(f"MDTM {filename}")
        return datetime.strptime(resp[4:], "%Y%m%d%H%M%S")
    except:
        return None

def should_download(local_path, remote_mtime):
    """Determine if download should proceed"""
    if not os.path.exists(local_path):
        return True
    
    local_mtime = datetime.fromtimestamp(os.path.getmtime(local_path))
    return remote_mtime > local_mtime

with FTP(base_url) as ftp:
    ftp.login()
    ftp.set_pasv(True)
    
    current_date = start_date.replace(day=1)
    while current_date <= end_date:
        date_str = current_date.strftime("%Y%m")
        data_status = "preliminary" if current_date >= cutoff_date else "stable"
        
        if data_status == "preliminary":
            print(f"Warning: {date_str} data is preliminary (may change within {stabilization_period} months)")
            
        for var in variables:
            remote_dir = f'/monthly/{var}/{current_date.year}/{date_str}'
            filename = f'PRISM_{var}_stable_4kmM3_{date_str}_bil.zip'
            local_dir = os.path.join(output_dir, var, date_str)
            local_path = os.path.join(local_dir, filename)
            
            try:
                ftp.cwd(remote_dir)
                remote_mtime = get_remote_mtime(ftp, filename)
                
                if not remote_mtime:
                    print(f"Skipping unavailable file: {filename}")
                    continue
                
                os.makedirs(local_dir, exist_ok=True)
                
                if should_download(local_path, remote_mtime):
                    print(f"Downloading {filename} (modified: {remote_mtime})")
                    with open(local_path, 'wb') as f:
                        ftp.retrbinary(f'RETR {filename}', f.write)
                else:
                    print(f"Up-to-date file exists: {filename}")
                    
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")
                continue
        
        # Move to next month
        current_date = current_date + timedelta(days=32)
        current_date = current_date.replace(day=1)

print("Download complete! Important notes:")
print("- Data less than 6 months old is preliminary and may change")
print("- Consider re-running periodically to catch updates")
print("- Final data becomes stable after 6 months")