# muFFTTO/jialing1105.py

def jialing1105_add(a, b):
    return a + b\



def edge_detection(input_field, **kwargs):
    """
    wraper function for all the inplemented edge detection functions
    """
    if kwargs has instance 'laplace':
        output_field= edge_detection_lapalce(input_field=input_field)

    elif kwargs has instance 'jacobi'
       output_field= edge_detection_jacobi(input_field=input_field)
    
    
    retrun output_field


def edge_detection_jacobi(input_field, **kwargs):

    if 'jacobi' not in kwargs:
        raise ValueError("kwargs does not contain 'jacobi'")
    if 'b' not in kwargs:
        raise ValueError("kwargs does not contain 'k' ")


    jac = kwargs['jacobi'] 


    
    return output_field

