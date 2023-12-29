"""
Setup script for readthedocs
"""
if __name__ in "__main__":
    import os
    import pypsg
    
    psg_key = os.environ.get('PSG_KEY')
    pypsg.settings.save_settings(api_key=psg_key)
