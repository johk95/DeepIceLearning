import os
import h5py
import numpy as np
import itertools

mycol = [(0,113/255.0,187/255.0),(89/255.0,186/255.0,115/255.0)]

# Define a context manager to suppress stdout and stderr.
class suppress_stdout_stderr(object):
    '''
    A context manager for doing a "deep suppression" of stdout and stderr in 
    Python, i.e. will suppress all print, even if the print originates in a 
    compiled C/Fortran sub-function.
       This will not suppress raised exceptions, since exceptions are printed
    to stderr just before a script exits, and after the context manager has
    exited (at least, I think that is why it lets exceptions through).      

    '''
    def __init__(self):
        # Open a pair of null files
        self.null_fds =  [os.open(os.devnull,os.O_RDWR) for x in range(2)]
        # Save the actual stdout (1) and stderr (2) file descriptors.
        self.save_fds = (os.dup(1), os.dup(2))

    def __enter__(self):
        # Assign the null pointers to stdout and stderr.
        os.dup2(self.null_fds[0],1)
        os.dup2(self.null_fds[1],2)

    def __exit__(self, *_):
        # Re-assign the real stdout/stderr back to (1) and (2)
        os.dup2(self.save_fds[0],1)
        os.dup2(self.save_fds[1],2)
        # Close the null files
        os.close(self.null_fds[0])
        os.close(self.null_fds[1])
        
        
def get_filenames(input_arg):
    """
    returns a list of filenames that were specified by the user with the --input argument. the sepcification can happen in these
    formats:
    'all': all files
    'highE', 'lowE': all high energy or low energy files
    'h{list of numbers}', 'l[list of numbers]': specifies which high or low energy files shall be used. example: h02l0 uses high[[0,2]] and l[0].
    'filename[:filename[...]]': directly given filenames joined by colon (:). These files must lie in config.cfg.dataenc_folder.trainnig_data
    
    Parameters
    ----------
    input_arg: str
        string as given by user (with --input argument)
        
    Returns
    -------
    list:
        list of input files, each given by the filename 110{29}_0{0-6}000-0{0-6}999\.h5
    """
    low_files = ['11029_00000-00999.h5','11029_01000-01999.h5','11029_02000-02999.h5',
                 '11029_03000-03999.h5','11029_04000-04999.h5','11029_05000-05999.h5']
    high_files = ['11069_00000-00999.h5','11069_01000-01999.h5','11069_02000-02999.h5',
                  '11069_03000-03999.h5','11069_04000-04999.h5','11069_05000-05999.h5','11069_06000-06999.h5']

    if input_arg.lower() == 'all':
        return low_files + high_files
    elif input_arg.lower() in 'lowe':
        return low_files
    elif input_arg.lower() in 'highe':
        return high_files
    elif input_arg.lower() in ['highe_reduced','high_reduced']: #same as highE, but only three files
        return high_files[0:3]
    elif len(input_arg) > 0 and (input_arg[0] == 'h' or input_arg[0] == 'l') and input_arg[1].isdigit():
        inputs = {k:v for k, v in zip(*[iter(["".join(x) for _, x in itertools.groupby(input_arg, key=str.isdigit)])]*2)}
        files = []
        if 'h' in inputs:
            files = [high_files[i] for i in map(int, list(inputs['h']))]
        if 'l' in inputs:
            files = filter(lambda s: len(s) > 0, files) + [low_files[i] for i in map(int, list(inputs['l']))]
        return files
    else:
        return input_arg.split(':')

        
def read_files(input_files, data_location, using='time', virtual_len=-1, printfilesizes=False):
    """
    Reads datasets from given input files and returns them as numpy arrays. 
    
    Parameters
    ----------
    input_files: list
        list of filenames. Files must lie in `data_location`/training_data/
    data_location: str
        path to the files enclosing folder (2 levels up, data lies in `data_location`/training_data/)
    using: str
        either "time" or "charge". Specifies which kind of data one wants to extract from the raw datasets. Defaults to "time"
    virtual_len: int
        optional parameter for rare cases where you want to bound the number of used datasets from each file.
    printfilesizes: bool
        if set to True, this function will print the length of (i.e. the number of datasets in) each given input file. returns 
        empty datasets.
        
    Returns
    -------
    list:
        input_data, list of the datasets in each input file(about 200'000 per highE file or 900'000 per 
        lowE file). in shape (len(input_files), 900'000, 20, 10, 60, 1)
    list:
        out_data, list of output values (reco_vals) for each input file. in shape (len(input_files), 900'000, 5)
    list:
        file_len, list of each input files length
    """
    input_data = []
    out_data = []
    file_len = []
    
  
    if printfilesizes:
        input_files = sorted(input_files)
        
    for run, input_file in enumerate(input_files):
        data_file = os.path.join(data_location, 'training_data/{}'.format(input_file))
  
        #print data_file, h5py.File(data_file,'r').keys()
        if virtual_len == -1:
            data_len = len(h5py.File(data_file,'r')[using])
        else:
            data_len = virtual_len
            print('Only use the first {} Monte Carlo Events'.format(data_len))
        if printfilesizes:
            print "{:10d}   {}".format(data_len, input_file)
        else:
            input_data.append(h5py.File(data_file, 'r')[using])
            out_data.append(h5py.File(data_file, 'r')['reco_vals'])
            file_len.append(data_len)
            #print type(input_data), type(input_data[-1]), type(input_data[-1][0]) #== list, h5py.Dataset, ndarray
            #print input_data[-1].shape #= (970452, 1, 21, 21, 51)
            
    return input_data, out_data, file_len


def zenith_to_binary(zenith, cosined = False):
    """
    returns boolean values for the zenith (0 or 1; up or down) and preserves input format.
    """
    if type(zenith) == np.ndarray:
        ret = np.copy(zenith)
        if cosined:
            ret = np.arccos(ret)
        ret[ret < 1.5707963268] = 0.0
        ret[ret > 1] = 1.0
        return ret
    if isinstance(zenith, float) or isinstance(zenith, int):
        if cosined:
            zenith = np.arccos(zenith)
        return 1.0 if zenith > 1.5707963268 else 0.0
    if isinstance(zenith, list):
        ret = np.array(zenith)
        if cosined:
            ret = np.arccos(ret)
        ret[ret < 1.5707963268] = 0.0
        ret[ret > 1] = 1.0
        return ret.tolist()
    
        
def preprocess(data, replace_with=1, normalize=True):
    """
    This function normalizes the finite values of input data to the interval [0,1] and 
    replaces all infinity-values with replace_with (defaults to 1).
    
    Parameters
    ----------
    data: list or ndarray
    
    replace_with: value that all np.inf's should be replaced with
    
    normalize: whether the data should be normalized or not.
    
    Returns
    -------
    ndarray
        A copy of the input data in range [0,1] and all np.inf replaced by replace_with.
    """
    ret = np.copy(data)
    if normalize:
        time_np_arr_max = np.max(ret[ret != np.inf])
        time_np_arr_min = np.min(ret)
        ret = (ret - time_np_arr_min) / (time_np_arr_max - time_np_arr_min)
    ret[ret == np.inf] = replace_with
    return ret

def fake_preprocess(data, replace_with=0, normalize=False):
    """
    This function is a helper if data has not to be preprocessed (i.e. all values stay as they are). This is an identity function
    and returns data.
    
    Parameters
    ----------
    data: object
    
    Returns
    -------
    object
        same as input data.
    """
    return data



def figsize(scale,scale_height=None):
    fig_width_pt = 360.0                          # Get this from LaTeX using \the\textwidth
    inches_per_pt = 1.0/72.27                       # Convert pt to inch
    golden_mean = (np.sqrt(5.0)-1.0)/2.0            # Aesthetic ratio (you could change this)
    if scale_height == None or type(scale_height) not in (int,float):
        scale_height = golden_mean
    height_scale = scale_height
    fig_width = fig_width_pt*inches_per_pt*scale    # width in inches
    fig_height = fig_width*height_scale              # height in inches
    fig_size = [fig_width,fig_height]
    return fig_size

def savefig(filename, folder="../plots"):
    plt.savefig(os.path.join(folder,'{}.pgf'.format(filename)))
    plt.savefig(os.path.join(folder,'{}_pdf.pdf'.format(filename)))

def get_plotted_x(x_hist, bins):
    ranges = float(x_hist[-1] - x_hist[0])
    ret = x_hist[:-1]+ranges/(bins-1)/2
    ret[0]= x_hist[0]
    ret[-1] = x_hist[-1]
    return ret

def get_xy_hist(out, res, bins, cosined = True, verbose=False):
    if np.min(out) < 0:
        out = np.arccos(out)
    if cosined:
        out = np.cos(out)
        x_hist = np.linspace(-1,1,bins)
    else:
        x_hist = np.linspace(0,np.pi,bins)
    y_hist = np.zeros((bins-1))
    cor, summe = 0, 0
    for i in range(len(x_hist)-1):
        indizes = np.bitwise_and(out >= x_hist[i], out < x_hist[i+1])
        correct = np.sum(np.round(res[indizes]) == zenith_to_binary(out[indizes],cosined=cosined))
        if verbose:
            print correct, np.sum(indizes), np.arccos(x_hist[i]) if cosined else x_hist[i] * 180/math.pi 
        cor += correct
        summe += np.sum(indizes)
        y_hist[i] = float(correct)/np.sum(indizes)
    print "{:.2f}%".format(100 * cor / float(summe))
    return x_hist, y_hist
    
def get_xy_zenith_hist(zenith_out, bins, cosined = True):
    x_hist = np.linspace(-1,1,bins) if cosined else np.linspace(0,np.pi,bins)
    y_zenith_hist = np.zeros((bins-1))
    hz = np.cos(zenith_out) if cosined else zenith_out
    for i in range(len(x_hist)-1):
        y_zenith_hist[i] = np.sum((hz >= x_hist[i]) & (hz < x_hist[i+1]))
    y_zenith_hist /= ((x_hist[1]-x_hist[0]) * np.sum(y_zenith_hist))
    return x_hist, y_zenith_hist