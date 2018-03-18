/*
%function to compute the rank-one SFPCA solution
%L1 sparse penalties
%for fixed regularization params
%%%%%%%%%%%%%%%%%%%%%%%%
%inputs:
%x: n x p data matrix 
%lamu: sparsity parameter for u
%lamv: sparsity parameter for v
%alphau: smoothness parameter for u
%alphav: smoothness parameter for v
%Omegu: n x n positive semi-definite matrix for roughness penalty on u
%Omegv: p x p positive semi-definite matrix for roughness penalty on v
%startu: n  vector of starting values for U;  if startu=0, then
%algorithm initialized to the rank-one SVD solution
%startv: p vector of starting values for V;  if startv=0, then
%algorithm initialized to the rank-one SVD solution
%posu: non-negativity indicator - posu = True imposes non-negative
%constraints on u, posu = False otherwise
%posv: non-negativity indicator - posv = True imposes non-negative
%constraints on v, posv = False otherwise
%maxit: maximum number of alternating regressions steps
%%%%%%%%%%%%%%%%
%outputs:
%U: n x 1 left SFPC
%V p x 1 right SFPC
%d: the associated singular value
%Xhat: the deflated residual matrix
*/

#include <iostream>

#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Sparse>
#include <eigen3/Eigen/Eigenvalues>
#include <eigen3/Eigen/SVD>



using namespace Eigen;
using namespace std;

MatrixXd soft_thr(MatrixXd a, double lam, bool pos){
	MatrixXd temp,b,SignMat;
	if(!pos){
		//In MATLAB u = sign(a).*max(abs(a) - lam,0);
		b = a.array().abs();
		b.array() -= lam;
		b.array() = b.array().max(0);
		SignMat = a.cwiseSign();
		temp = (SignMat.array() * b.array()).matrix();
		return temp;
	}
	else{
		//In MATLAB u = max(a - lam,0);
		a.array() -= lam;
		temp.array() = a.array().max(0);
		return temp;
	}
}
// Returns the 2-norm or maximum singular value of matrix a
double Norm(MatrixXd a){
	BDCSVD<MatrixXd> Svd(a, ComputeThinU | ComputeThinV);
	return (Svd.singularValues()).maxCoeff();
}

MatrixXd * sfpca_fixed(
	MatrixXd x,
	double lamu,
	double lamv,
	double alphau,
	double alphav,
	MatrixXd Omegu,
	MatrixXd Omegv,
	MatrixXd startu,
	MatrixXd startv,
	bool posu,
	bool posv,
	double maxit
	)
{
	MatrixXd *Answer =new MatrixXd[4];
	MatrixXd u,v,oldu,oldv,oldui,oldvi,utild,vtild;
	double n,p,d;
	n = x.rows();
	p = x.cols();
	SparseMatrix<double> Su(n,n) ,Sv(p,p);
    Su.setIdentity();
	Sv.setIdentity();
	Su += n*alphau*Omegu.sparseView();
	Sv += p*alphav*Omegv.sparseView();
	double Lu = MatrixXd(Su).eigenvalues().real().maxCoeff() + 0.1;
	double Lv = MatrixXd(Sv).eigenvalues().real().maxCoeff() + 0.1;
	double thr = 1e-6;
	MatrixXd Xhat = x;
	
	if(startu.sum()==0){
		BDCSVD<MatrixXd> svd(x, ComputeThinU | ComputeThinV);
		MatrixXd utemp,vtemp;
		utemp = svd.matrixU();
		vtemp = svd.matrixV();
		u = utemp.col(0);
		v = vtemp.col(0);
		d = (svd.singularValues()).maxCoeff();
	}
	else{
		u = startu;
		v = startv;
	}

	double indo=1,iter=0;
	while((indo>thr) && (iter<maxit)){
		oldu = u;
		oldv = v;
		double indu = 1;
		while(indu>thr){
			oldui = u;
			utild = u + (Xhat*v - Su*u)/Lu;
			u = soft_thr(utild, lamu/Lu, posu);
			double unorm = Norm(u);
			if(unorm>0){
				double mod = ((u.transpose())*Su*u)(0,0);
				mod = std::sqrt(mod);
				u = u/mod;
			}
			else{
				u = MatrixXd::Zero(n,1);
			}
			indu = Norm(u-oldui)/Norm(oldui);
		}

		double indv = 1;
		while(indv>thr){
			oldvi = v;
			vtild = v + ((Xhat.transpose())*u - Sv*v)/Lv;
			v = soft_thr(vtild, lamv/Lv, posv);
			double vnorm = Norm(v);
			if(vnorm>0){
				double mod = ((v.transpose())*Sv*v)(0,0);
				mod = std::sqrt(mod);
				v = v/mod;
			}
			else{
				v = MatrixXd::Zero(p,1);
			}
			indv = Norm(v-oldvi)/Norm(oldvi);
		}

		indo = Norm(oldu-u)/Norm(oldu) + Norm(oldv-v)/Norm(oldv);
		iter++;
	}
	Answer[0] = u/Norm(u);
	Answer[1] = v/Norm(v);
	Answer[2] = (Answer[0].transpose())*Xhat*Answer[1];
	Answer[3] = Xhat - (Answer[2](0,0))*Answer[0]*(Answer[1].transpose());
	return Answer;
}
