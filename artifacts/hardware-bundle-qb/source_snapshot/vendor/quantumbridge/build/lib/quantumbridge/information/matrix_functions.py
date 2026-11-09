# Copyright 2026 QuantumBridge Contributors.
# Licensed under the Apache License, Version 2.0.
"""Bounded native complex powers using unitary Schur reduction and binomial ISS.

The scalar branch and matrix binomial identities are mathematical references:
https://nhigham.com/2020/12/08/what-is-a-fractional-matrix-power/
https://nhigham.com/2020/06/23/what-is-a-frechet-derivative/
This is an independent NumPy implementation, not upstream SDK source.
"""

import cmath
import math
from fractions import Fraction

import numpy as np

from quantumbridge.core.classical import UnsupportedFeature
from quantumbridge.core.gate_limits import current_gate_limits
from quantumbridge.execution import mps_arrays as arrays
from quantumbridge.execution.array_ownership import retain_reservation,remember_storage_ledger
from quantumbridge.execution.array_reservations import ArrayReservation
from quantumbridge.execution.resource_context import current_resources


_EPS=np.finfo(float).eps
_REAL_TYPES=(bool,int,float,np.bool_,np.int8,np.int16,np.int32,np.int64,
    np.uint8,np.uint16,np.uint32,np.uint64,np.longlong,np.ulonglong,
    np.float16,np.float32,np.float64,np.longdouble)


def _real(value,name):
    if not any(type(value) is kind for kind in _REAL_TYPES):
        raise TypeError(name+" must be a finite native real scalar.")
    if type(value) is int or type(value) is bool or isinstance(value,np.integer):return int(value)
    value=float(value)
    if not math.isfinite(value):raise ValueError(name+" must be finite.")
    return value


class _Work:
    def __init__(self,size):
        self.value=0
        current_gate_limits().check("max_matrix_dimension",size)
        self.charge(size*size)

    def charge(self,count):
        self.value+=count
        current_gate_limits().check("max_decomposition_work",self.value)


def _norm(matrix):
    return max((sum(abs(complex(matrix[row,col])) for row in range(matrix.shape[0]))
                for col in range(matrix.shape[1])),default=0.)


def _finite(matrix):
    if any(not (math.isfinite(value.real) and math.isfinite(value.imag)) for value in matrix.flat):
        raise np.linalg.LinAlgError("Matrix power is outside the finite complex128 numerical range.")


def _identity(size):
    result=arrays.filled("operator_power_identity",(size,size),complex,0)
    result.flat[::size+1]=1
    return result


def _product(left,right,work):
    work.charge(left.shape[0]*left.shape[1]*right.shape[1])
    result=arrays.contracted(left,right)
    _finite(result)
    return result


def _eigen_pair(matrix,work):
    size=len(matrix);work.charge(size**3)
    scope=current_resources();ledger=None if scope is None else scope.ledger
    reservations=[ArrayReservation(ledger,"operator_power_eigenvalues",(size,),complex)]
    transferred=False
    try:
        reservations.append(ArrayReservation(ledger,"operator_power_eigenvectors",matrix.shape,complex))
        result=np.linalg.eig(matrix)
        for ticket,value in zip(reservations,result):ticket.attach(value)
        for ticket,value in zip(reservations,result):
            retain_reservation(ticket,value);remember_storage_ledger(value,ledger)
        transferred=True
        return result
    finally:
        if not transferred:
            for ticket in reservations:ticket.release()


def _schur(matrix,work):
    """Unitary eigenvector deflation, never inversion of an eigenvector matrix."""
    size=len(matrix);triangular=arrays.copied(matrix,category="operator_power_schur")
    basis=_identity(size)
    if all(triangular[row,col]==0 for row in range(size) for col in range(row)) and all(triangular[i,i]!=0 for i in range(size)):
        return triangular,basis
    for start in range(size-1):
        tail=triangular[start:,start:];width=len(tail)
        values,vectors=_eigen_pair(tail,work)
        index=0
        for i in range(1,width):
            if abs(values[i])<abs(values[index]):index=i
        vector=arrays.copied(vectors[:,index],category="operator_power_householder_vector")
        length=math.sqrt(sum(abs(complex(value))**2 for value in vector))
        if not math.isfinite(length) or length==0:raise np.linalg.LinAlgError("Invalid Schur deflation vector.")
        vector/=length
        phase=1 if vector[0]==0 else vector[0]/abs(vector[0])
        vector[0]+=phase
        length=math.sqrt(sum(abs(complex(value))**2 for value in vector));vector/=length
        reflection=arrays.owned("operator_power_householder",(width,width),complex)
        conjugate=arrays.conjugated(vector)
        np.multiply(vector[:,None],conjugate[None,:],out=reflection)
        reflection*=-2;reflection.flat[::width+1]+=1
        triangular[start:,:]=_product(reflection,triangular[start:,:],work)
        triangular[:,start:]=_product(triangular[:,start:],reflection,work)
        basis[:,start:]=_product(basis[:,start:],reflection,work)
        residual=sum(abs(triangular[row,start]) for row in range(start+1,size))
        if residual>128*_EPS*size*max(_norm(matrix),np.finfo(float).tiny):
            raise np.linalg.LinAlgError("Schur deflation did not attain its backward-error bound.")
        triangular[start+1:,start]=0
        del values,vectors,vector,conjugate,reflection
    reconstruction=_product(_product(basis,triangular,work),arrays.conjugated(basis).T,work)
    reconstruction-=matrix
    if _norm(reconstruction)>256*_EPS*size*max(_norm(matrix),np.finfo(float).tiny):
        raise np.linalg.LinAlgError("Schur reconstruction exceeds its backward-error bound.")
    return triangular,basis


def _scalar(value,p,cut):
    if value==0:
        if p<0:raise np.linalg.LinAlgError("A negative power requires a nonsingular operator.")
        return 1.+0j if p==0 else 0j
    try:
        rotated=value*cmath.exp(-1j*cut)
        return cmath.exp(p*(math.log(abs(rotated))+1j*(cmath.phase(rotated)+cut)))
    except (OverflowError,ValueError) as error:
        raise np.linalg.LinAlgError("Matrix power is outside the finite scalar numerical range.") from error


def _scalar_integer(value,n):
    if n<0:
        if value==0:raise np.linalg.LinAlgError("A negative power requires a nonsingular operator.")
        value=1/value;n=-n
    result=1.+0j
    while n:
        if n&1:result*=value
        n>>=1
        if n:value*=value
    return result


def _log1p(value):
    x,y=value.real,value.imag
    return complex(.5*math.log1p(2*x+x*x+y*y),math.atan2(y,1+x))


def _expm1(value):
    x,y=value.real,value.imag
    return complex(math.expm1(x)*math.cos(y)-2*math.sin(y/2)**2,math.exp(x)*math.sin(y))


def _divided_power(a,b,loga,logb,p):
    if a==b:return p*cmath.exp((p-1)*loga)
    difference=loga-logb
    if abs(difference)<.5:
        difference=_log1p((a-b)/b)
        return cmath.exp(p*logb)*_expm1(p*difference)/(a-b)
    return (cmath.exp(p*loga)-cmath.exp(p*logb))/(a-b)


def _correct_bands(output,original,logs,p):
    for row in range(len(output)):
        output[row,row]=cmath.exp(p*logs[row])
        if row+1<len(output) and original[row,row+1]!=0:
            output[row,row+1]=original[row,row+1]*_divided_power(
                original[row,row],original[row+1,row+1],logs[row],logs[row+1],p)


def _exact_repeated_pair(matrix,p,cut,assume,work):
    """Certify a repeated 2x2 spectrum from the exact binary input scalars.

    A small floating discriminant is not a certificate: on a logarithm cut it
    can conceal two genuinely different branches. This constant-size rational
    identity check does not approximate or coalesce any nonzero discriminant.
    """
    if len(matrix)!=2:return None
    work.charge(128)
    parts=[(Fraction.from_float(float(z.real)),Fraction.from_float(float(z.imag))) for z in matrix.flat]
    a,b,c,d=parts
    real,imag=(a[0]-d[0])/2,(a[1]-d[1])/2
    discriminant=(real*real-imag*imag+b[0]*c[0]-b[1]*c[1],
                  2*real*imag+b[0]*c[1]+b[1]*c[0])
    if discriminant!=(0,0):return None
    center=complex(float((a[0]+d[0])/2),float((a[1]+d[1])/2))
    if center==0:raise np.linalg.LinAlgError("Fractional power is undefined on this singular Jordan block.")
    if assume:raise ValueError("assume_unitary requires a numerically normal operator.")
    scope=current_resources()
    if scope is not None:scope.ledger.unavailable("operator_power_scalar_identity_workspace",
        "A constant-size exact binary-scalar certificate uses bounded Python integers; Python allocator/RSS is not observed.")
    scalar=_scalar(center,p,cut)
    result=arrays.copied(matrix,category="operator_power_exact_repeated_pair")
    result.flat[::3]-=center
    result/=center;result*=p;result*=scalar
    result.flat[::3]+=scalar
    _finite(result)
    return result


def _check_branch_sensitivity(matrix,logs,work):
    """Screen coupled Schur pairs whose branch is unresolved by backward error."""
    size=len(matrix)
    resolution=256*_EPS*size*_norm(matrix)
    for row in range(size):
        for col in range(row+1,size):
            work.charge(1)
            if abs(logs[row].imag-logs[col].imag)<=math.pi:continue
            coupling=abs(matrix[row,col])
            uncertainty=8*math.sqrt(resolution*coupling)
            gap=abs(matrix[row,row]-matrix[col,col])
            if coupling and gap<=uncertainty:
                raise np.linalg.LinAlgError(
                    "Matrix power branch-sensitive Schur cluster is unresolved: "
                    f"eigenvalue gap {gap:.6g} <= sensitivity screen {uncertainty:.6g} "
                    f"(coupling {coupling:.6g}, backward resolution {resolution:.6g}).")


def _triangular_fraction(matrix,p,work):
    size=len(matrix);logs=tuple(cmath.log(matrix[i,i]) for i in range(size))
    _check_branch_sensitivity(matrix,logs,work)
    reduced=arrays.copied(matrix,category="operator_power_root_input")
    identity=_identity(size);roots=0
    while True:
        delta=arrays.copied(reduced,category="operator_power_binomial_argument");delta-=identity
        radius=_norm(delta)
        if radius<=.25:break
        if roots>=32:raise np.linalg.LinAlgError("Matrix power square-root scaling did not converge.")
        roots+=1
        work.charge(size**3)
        root=arrays.filled("operator_power_triangular_root",matrix.shape,complex,0)
        for row in range(size):root[row,row]=cmath.exp(logs[row]/(2**roots))
        for distance in range(1,size):
            for row in range(size-distance):
                col=row+distance
                residual=reduced[row,col]-sum(root[row,k]*root[k,col] for k in range(row+1,col))
                if residual==0:continue
                denominator=root[row,row]+root[col,col]
                if abs(denominator)<=16*_EPS*max(abs(root[row,row]),abs(root[col,col])):
                    raise np.linalg.LinAlgError("Matrix power is numerically ill-conditioned near its branch cut.")
                root[row,col]=residual/denominator
        _finite(root);reduced=root
    # ||delta|| <= 1/4 bounds the tail of this absolutely convergent matrix series.
    for row in range(size):delta[row,row]=_expm1(logs[row]/(2**roots))
    radius=_norm(delta);result=_identity(size);term=_identity(size)
    for order in range(1,129):
        term=_product(term,delta,work);term*=(p-order+1)/order;result+=term
        if radius**(order+1)/(1-radius)<=_EPS/16:break
    else:raise np.linalg.LinAlgError("Matrix power binomial remainder did not converge.")
    _correct_bands(result,matrix,logs,p/(2**roots))
    for level in range(roots-1,-1,-1):
        result=_product(result,result,work)
        _correct_bands(result,matrix,logs,p/(2**level))
    _finite(result)
    return result


def _integer(matrix,n,work):
    size=len(matrix);bits=abs(n).bit_length()
    work.charge(size**3*(bits+abs(n).bit_count()))
    if n<0:
        inverse=arrays.owned("operator_power_inverse",matrix.shape,complex,lambda:np.linalg.inv(matrix))
        condition=_norm(matrix)*_norm(inverse)
        if not math.isfinite(condition) or condition>1/(256*_EPS*size):
            raise np.linalg.LinAlgError("Matrix inverse power is numerically ill-conditioned.")
        matrix=inverse;n=-n
    result=arrays.matrix_power(matrix,n)
    _finite(result)
    return result


def _disconnected_blocks(matrix,work):
    size=len(matrix);work.charge(size*size)
    parent=list(range(size))
    def root(index):
        while parent[index]!=index:
            parent[index]=parent[parent[index]];index=parent[index]
        return index
    for row in range(size):
        for col in range(row+1,size):
            if matrix[row,col]!=0 or matrix[col,row]!=0:
                left,right=root(row),root(col)
                if left!=right:parent[right]=left
    components={}
    for index in range(size):components.setdefault(root(index),[]).append(index)
    return list(components.values())


def _fractional(matrix,n,cut,assume,work):
    size=len(matrix);scale=max(abs(complex(value)) for value in matrix.flat)
    if scale==0:
        if n<0:raise np.linalg.LinAlgError("A negative power requires a nonsingular operator.")
        return arrays.filled("operator_power_zero",matrix.shape,complex,0)
    exact_pair=_exact_repeated_pair(matrix,n,cut,assume,work)
    if exact_pair is not None:return exact_pair
    components=_disconnected_blocks(matrix,work)
    if len(components)>1:
        result=arrays.filled("operator_power_disconnected_blocks",matrix.shape,complex,0)
        for indices in components:
            if len(indices)==1:
                row=indices[0];result[row,row]=_scalar(complex(matrix[row,row]),n,cut)
                continue
            block=arrays.owned("operator_power_component",(len(indices),len(indices)),complex)
            for row,source_row in enumerate(indices):
                for col,source_col in enumerate(indices):block[row,col]=matrix[source_row,source_col]
            powered=_fractional(block,n,cut,assume,work)
            for row,dest_row in enumerate(indices):
                for col,dest_col in enumerate(indices):result[dest_row,dest_col]=powered[row,col]
            del block,powered
        _finite(result)
        return result
    normalized=arrays.divided(matrix,scale)
    triangular,basis=_schur(normalized,work)
    if assume:
        adjoint=arrays.conjugated(normalized).T
        first=_product(normalized,adjoint,work);second=_product(adjoint,normalized,work);first-=second
        offdiagonal=max((sum(abs(triangular[row,col]) for row in range(col)) for col in range(size)),default=0.)
        if (_norm(first)>64*_EPS*size*max(_norm(normalized)**2,np.finfo(float).tiny)
                or offdiagonal>64*_EPS*size*_norm(triangular)):
            raise ValueError("assume_unitary requires a numerically normal operator.")
    if any(0<abs(triangular[i,i])<=128*_EPS*size*_norm(triangular) for i in range(size)):
        raise np.linalg.LinAlgError("Matrix power has a numerically unresolved near-zero eigenvalue.")
    triangular*=cmath.exp(-1j*cut)
    diagonal=tuple(triangular[i,i] for i in range(size))
    zeros=0
    while zeros<size and diagonal[zeros]==0:zeros+=1
    if any(value==0 for value in diagonal[zeros:]):
        raise np.linalg.LinAlgError("Singular Schur block could not be separated reliably.")
    if zeros and (n<0 or any(triangular[row,col]!=0 for row in range(zeros) for col in range(zeros))):
        raise np.linalg.LinAlgError("Fractional power is undefined on this singular Jordan block.")
    block=triangular[zeros:,zeros:]
    integer=math.trunc(n);fraction=n-integer
    fractional=_triangular_fraction(block,fraction,work)
    block_power=_product(_integer(block,integer,work),fractional,work)
    powered=arrays.filled("operator_power_triangular",matrix.shape,complex,0)
    powered[zeros:,zeros:]=block_power
    if zeros:
        inverse=_integer(block,-1,work)
        powered[:zeros,zeros:]=_product(triangular[:zeros,zeros:],_product(inverse,block_power,work),work)
    try:powered*=cmath.exp(n*(math.log(scale)+1j*cut))
    except (OverflowError,ValueError) as error:raise np.linalg.LinAlgError("Matrix power scaling overflow.") from error
    return _product(_product(basis,powered,work),arrays.conjugated(basis).T,work)


def matrix_power(matrix,n,branch_cut_rotation,assume_unitary,*,_work=None):
    n=_real(n,"Operator exponent");cut=_real(branch_cut_rotation,"Branch cut rotation")
    try:cut=float(cut)
    except OverflowError as error:raise ValueError("Branch cut rotation must be finite.") from error
    if not math.isfinite(cut):raise ValueError("Branch cut rotation must be finite.")
    if type(assume_unitary) is not bool and type(assume_unitary) is not np.bool_:
        raise TypeError("assume_unitary must be a native boolean.")
    if not issubclass(type(matrix),np.ndarray):raise UnsupportedFeature("operator_power_numeric_buffer")
    matrix=np.ndarray.view(matrix,np.ndarray)
    if matrix.dtype.kind not in "biufc":raise UnsupportedFeature("operator_power_numeric_dtype")
    if matrix.ndim!=2 or matrix.shape[0]!=matrix.shape[1] or not matrix.shape[0]:
        raise ValueError("Operator powers require a nonempty square matrix.")
    size=len(matrix);work=_Work(size) if _work is None else _work
    current_gate_limits().check("max_matrix_dimension",size)
    work.charge(size*size) if _work is not None else None
    _finite(matrix)
    integral=type(n) is int or n.is_integer()
    if integral:
        n=int(n);work.charge(abs(n).bit_length()*size**3)
    diagonal=all(matrix[row,col]==0 for row in range(size) for col in range(size) if row!=col)
    scope=current_resources();ledger=None if scope is None else scope.ledger
    numerical=(not integral or n<0) and not diagonal
    workspace=16*size*size+32*size if numerical else 0
    if scope is not None:
        destinations=1 if diagonal or integral and n in (0,1) else 48 if numerical else 4
        scope.ledger.preflight(memory_bytes=(workspace+destinations*size*size)*16)
        if numerical:scope.ledger.unavailable("operator_power_lapack_allocator_workspace",
            "A conservative numeric workspace envelope is reserved; hidden vendor LAPACK allocator/RSS is not observed.")
    with ArrayReservation(ledger,"operator_power_workspace_envelope",(workspace,),complex):
        if diagonal:
            result=arrays.filled("operator_power_diagonal",matrix.shape,complex,0)
            for i in range(size):
                if integral:
                    result[i,i]=_scalar_integer(complex(matrix[i,i]),n)
                else:result[i,i]=_scalar(complex(matrix[i,i]),n,cut)
        elif integral and n==0:result=_identity(size)
        elif integral and n==1:result=arrays.copied(matrix,dtype=complex,category="operator_power_input_copy")
        else:
            copied=arrays.copied(matrix,dtype=complex,category="operator_power_input_copy")
            with np.errstate(over="ignore",invalid="ignore",divide="ignore"):
                result=_integer(copied,n,work) if integral else _fractional(copied,n,cut,bool(assume_unitary),work)
        _finite(result)
        return result


def unitary_power_jet(matrix, first, second, exponent, *, branch_cut_rotation=0, _work=None):
    """Return f(U), (f(U))', (f(U))'' without assuming [U,U'] is zero.

In the unitary Schur frame, apply the existing analytic matrix function to
[[T,E,F/2],[0,T,E],[0,0,T]]. Its (0,1) block is L_f(T,E), and twice (0,2)
is L_f(T,F)+D2_f(T)[E,E]. No eigenvector derivatives or finite differences.
The enlarged triangular operator shares one work budget with this wrapper.
"""
    return _unitary_power_blocks(matrix, (first, second), exponent, branch_cut_rotation, _work)


def unitary_power_mixed_jet(matrix, first, other, mixed, exponent, *, branch_cut_rotation=0, _work=None):
    """Return f(U), D_a f(U), D_b f(U), D_ab f(U) with actual mixed source.

    In a unitary Schur frame, the (0,3) block of f([[T,E,G,F],
    [0,T,0,G],[0,0,T,E],[0,0,0,T]]) contains L_f(T,F) and both
    ordered E/G paths. Directions are evaluated without a stored Hessian.
    """
    return _unitary_power_blocks(matrix, (first, other, mixed), exponent, branch_cut_rotation, _work)


def _unitary_power_blocks(matrix, directions, exponent, branch_cut_rotation, _work):
    from quantumbridge.execution.array_reservations import numeric_shape, numeric_array
    inputs = (matrix, *directions)
    shapes = [numeric_shape(value) for value in inputs]
    if (len(shapes[0]) != 2 or not shapes[0][0] or shapes[0][0] != shapes[0][1]
            or any(shape != shapes[0] for shape in shapes[1:])):
        raise ValueError("Unitary power jets require matching nonempty square matrices.")
    size = shapes[0][0]
    blocks = len(inputs)
    exponent = _real(exponent, "Power-jet exponent")
    cut = float(_real(branch_cut_rotation, "Power-jet branch rotation"))
    current_gate_limits().check("max_matrix_dimension", blocks*size)
    work = _Work(blocks*size) if _work is None else _work
    if _work is not None: work.charge((blocks*size)**2)
    scope = current_resources()
    ledger = None if scope is None else scope.ledger
    if scope is not None:
        scope.ledger.preflight(memory_bytes=16*128*(blocks*size)**2)
        scope.ledger.unavailable("power_jet_lapack_workspace",
            "Native outputs/workspace are reserved; hidden LAPACK allocator/RSS is not observed.")
    with ArrayReservation(ledger, "power_jet_external_workspace", (max(8,2*blocks)*size*size,), complex):
        matrix, *directions = [numeric_array(value, category="power_jet_input") for value in inputs]
        for value in (matrix, *directions): _finite(value)
        adjoint = arrays.conjugated(matrix).T
        residual = _product(adjoint, matrix, work)
        residual.flat[::size+1] -= 1
        if _norm(residual) > 512*_EPS*size:
            raise ValueError("Power-jet base must be unitary.")
        triangular, basis = _schur(matrix, work)
        integral = isinstance(exponent, int) or exponent.is_integer()
        if not integral:
            phases = []
            for index in range(size):
                value = complex(triangular[index,index])*cmath.exp(-1j*cut)
                phase = cmath.phase(value)
                phases.append(phase)
                if abs(abs(phase)-math.pi) <= 512*_EPS*size:
                    raise np.linalg.LinAlgError("Fractional power derivative is unresolved at its branch cut.")
            # Across the cut, near-coincident eigenvalues represent different
            # branches. Second jets have cubic separation sensitivity: tiny
            # backward residuals alone cannot certify a reliable derivative.
            work.charge(size*size)
            for row in range(size):
                for col in range(row):
                    if abs(phases[row]-phases[col]) <= math.pi:
                        continue
                    separation = abs(complex(triangular[row,row])-complex(triangular[col,col]))
                    uncertainty = (64*_EPS*size**2*(1+abs(exponent))**2
                        *abs(math.sin(math.pi*exponent))/max(separation,np.finfo(float).tiny)**3)
                    if uncertainty > math.sqrt(_EPS):
                        coupling = max((_norm(value) for value in directions), default=0.)
                        diagnostics = dict(gap=separation, coupling=coupling,
                            resolution=512*_EPS*size, screen=uncertainty,
                            threshold=math.sqrt(_EPS), kind="conservative_cubic_roundoff_screen")
                        error = np.linalg.LinAlgError(
                            "Matrix-power derivative is numerically ill-conditioned across its branch cut; "
                            "the cubic spectral-separation roundoff screen cannot certify this jet. "
                            f"gap={separation:.8g}, coupling_bound={coupling:.8g}, "
                            f"resolution={diagnostics['resolution']:.8g}, screen={uncertainty:.8g}.")
                        error.diagnostics = diagnostics
                        raise error
        inverse_basis = arrays.conjugated(basis).T
        transformed = [_product(_product(inverse_basis, direction, work), basis, work) for direction in directions]
        block = arrays.filled("power_jet_triangular_block", (blocks*size,blocks*size), complex, 0)
        for offset in range(0, blocks*size, size): block[offset:offset+size,offset:offset+size] = triangular
        e = transformed[0]
        block[:size,size:2*size] = e
        if blocks == 3:
            block[size:2*size,2*size:] = e
            np.multiply(transformed[1],.5,out=block[:size,2*size:])
        else:
            block[:size,2*size:3*size] = transformed[1]
            block[:size,3*size:] = transformed[2]
            block[size:2*size,3*size:] = transformed[1]
            block[2*size:3*size,3*size:] = e
        result = matrix_power(block, exponent, cut, False, _work=work)
        outputs = []
        for index in range(blocks):
            selected = result[:size,index*size:(index+1)*size]
            value = _product(_product(basis, selected, work), inverse_basis, work)
            if blocks == 3 and index == 2: value *= 2
            outputs.append(value)
        return tuple(outputs)
