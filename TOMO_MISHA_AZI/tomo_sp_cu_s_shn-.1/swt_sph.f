c********************************************************************
c*     Program is designed for tomographic reconsts                 *
c*     of surface wave phase velocities ON SPHERICAL SURFACE        *
c*     from the data on path velocities                             *
c*     SOLUTION IS DETERMINED ON THE SPHERE DIRECTLY                *
c********************************************************************
      SUBROUTINE SWTSPH
      IMPLICIT NONE
      include "tomo.h"
      include "line.h"
      include "linea.h"
c-------------------------------------------------------------
      real*4    f((NRAZ+NRAZA*(NAZIPL-1))*(NRAZ+NRAZA*(NAZIPL-1))),
     +          dazi(NRAZ+NRAZA),d2azi(NRAZ+NRAZA)
      real*4    fsl(NRAZ+NRAZA*(NAZIPL-1)),denf(NRAZ)
      real*4    ff(NRAZ+NRAZA*(NAZIPL-1)),dens(NRAZ+NRAZA*(NAZIPL-1)),
     +          gis(IAZIM,(NRAZ+NRAZA))
      real*4    denfs(NRAZ),denss(NRAZ+NRAZA*(NAZIPL-1))
      real*4    den1f(NRAZ),den2f(NRAZ),den1fs(NRAZ),den2fs(NRAZ)
      real*4    value(5),wval(5),wval1(5),wval2(5),gval(5),gval2(5)
      real*4    dvalue(5),alpha,alpha1
      real*4    azmax,s2,gsum,gise,wexp1,wnorm,RBIMOD
      real*4    rbimod1,rcond,rrr,rrr4,rrrr,rrrr4,s,s1,s2s,s2sn,dm,
     +          sre,sre1,sre_norm,uuu,value1,vamp1,vamp2,vph1,
     +          vph2,vvv,w2w,wsum,wsum1,wsum2,ww,ww1,ww2
      integer*4 ipvt(NRAZ+NRAZA*(NAZIPL-1)),
     +          lidd(NRAZ+NRAZA*(NAZIPL-1)),lidd1(NRAZ+NRAZA*(NAZIPL-1))
      integer*4 i,i1,i2,i3,iccc,ierr,ihome,ii,iou,
     +          j,jj,k,kk,l,ll,lii,lj,nff,nlidd,nmAL,ns2s,
     +          iswt,nlidd1,kkk,n1,n2,nmgis,ljj,ip,jp
      real*4    fff0,fff1
c-------------------------------------------------------------
      real*8 xh,yh,zh,xk,yk,tc00
      character*4 type,type1
c-------------------------------------------------------------
      real*8 x1(3),x2(3),xn(3),e(3),delta,step,dlen,st1,st2,firot
      real*8 sigs2s,sigv2v,sigw2w,sss,sxy,xk1(NRAZ),yk1(NRAZ)
      real*8 duaz
      character*80 pstr
      character*160 fdres,fdres1,fdres2,cmd
      character*8 spid,spida
      integer*4 npid,getpid,lnblnk
C--------- Creation cellular net ------------------------------------
      if(ldresol.or.lresol) then
        npid=getpid()
        write(spida,'(i8)') npid
        do i=1,8
          if(spida(i:i).ne.' ') goto 777
        enddo
 777    spid=spida(i:8)
        l=lnblnk(spid)
        k=lnblnk(fname6)
        fdres= fname6(1:k)//'1.'//spid(1:l)
        fdres1=fname6(1:k)//'2.'//spid(1:l)
        fdres2=fname6(1:k)//'3.'//spid(1:l)
      endif
      type=re_la
      type1=gr_pha
      CALL TRASS(n_pnt,na_pnt)
      if(ianiz.ne.0) then
        write(8,*) 'n_pnt1 = ',n_pnt,' na_pnt = ',na_pnt
        write(*,*) 'n_pnt1 = ',n_pnt,' na_pnt = ',na_pnt
      else
        write(8,*) 'n_pnt1 = ',n_pnt
        write(*,*) 'n_pnt1 = ',n_pnt
      endif
      write(*,*)'Cells: ',nall,', nm=',nm,', nd+1=',nd+1,', aniz=',0
      write(8,*)'Cells: ',nall,', nm=',nm,', nd+1=',nd+1,', aniz=',0
      if(ianiz.ne.0) then
        write(*,*)'Cells: ',nalla,', nma=',nma,', nda+1=',nda+1,
     +  ', aniz=',ianiz+istrip
        write(8,*)'Cells: ',nalla,', nma=',nma,', nda+1=',nda+1,
     +  ', aniz=',ianiz+istrip
        wnorm=FLOAT(2*n_pnt+1)/FLOAT(2*na_pnt+1)
        wexp1=wexp*wnorm
        write(8,1000) wexp,wexp1
        write(*,1000) wexp,wexp1
      else
        write(8,1002) wexp
        write(*,1002) wexp
      endif
 1000 format("Normalization factors: Isotr -",F9.5,", Aniz:",F9.5)
 1002 format("Normalization factors: Isotr -",F9.5)
C--------- Constants definition-------------------------------------
      KOD=0
      if(nm.gt.NRAZ) then
        KOD=1
        write(*,*)'(R0012) SIZE OF NRAZ IS TOO SMALL',NRAZ,
     +  ' SHOULD BE:',nm
        write(8,*)'SIZE OF NRAZ is TOO SMALL',NRAZ,
     +  ' SHOULD BE:',nm
        STOP
      endif
      if(ianiz.ne.0) then
        if(nma.gt.NRAZA) then
          KOD=1
          write(*,*)'(R0013) SIZE OF NRAZA IS TOO SMALL',NRAZA,
     +    ' SHOULD BE:',nma
          write(8,*)'SIZE OF NRAZA is TOO SMALL',NRAZA,
     +    ' SHOULD BE:',nma
          STOP
        endif
      endif
      NAZIP=ianiz*2+1
      alpha1=alsi(1,1)
      alpha= alsi(1,2)
      nmAL=nm+nma*(NAZIP-1)
      nmgis=max0(nm,nma)
c
c   Setup arrays and variables to zero
c
      nff=0
      do i=1,nmAL*nmAL
        f(i)=0.0
      enddo
      do i=1,nmgis
      do j=1,IAZIM
        gis(j,i)=0.0
      enddo
      enddo
      if(ianiz.ne.0) then
        do i=1,nma
          dens(i)=0.0
          denss(i)=0.0
        enddo
      endif
      do i=1,nm
        denf(i)=0.0
        denfs(i)=0.0
      enddo
      do i=1,nmAL
        fsl(i)=0.0
      enddo
      sigs2s=0.0d0
      sigv2v=0.0d0
      sigw2w=0.0d0
      write(*,1001) n
      write(8,1001) n
 1001 format('    Number of Paths=',i7)
      n1=0
      n2=0
c
c  Compute matrix element for input rays.
c  There are two branches of algorithm.
c  The first one for Gaussian ray and another one for
c  Fresnel zone
c
      do 7 i=1,n
      ncoin0=-1
      if(IRAY2(i).eq.1) n1=n1+1
      if(IRAY2(i).eq.2) n2=n2+1
      CALL TO_CART(TE0(i),FI0(i),TE(i),FI(i),IRAY2(i),x1,x2,xn,delta)
      T(i)=DBLE(R)*delta/DBLE(T(i))
      nlidd=0
      lidd(1)=0
      do 1 j=1,nm
      denf(j)=0.0
      den1f(j)=0.0
      den2f(j)=0.0
    1 ff(j)=0.0
      tc00=0.0
      if(itomo.eq.0) then
c
c    compute Gaussian ray
c
      step=step0*drad
      iswt=0
      st1=0.0d0
      st2=step
      dlen=0.0d0
   72 if(st2.ge.delta) then
      st2=delta
      step=st2-st1
      iswt=1
      endif
      firot=(st2+st1)/2.0d0
      CALL RTURN(firot,xn,x1,e)
      call NORM(e,e)
      CALL ESTRACEN(lidd,nlidd,step,e,nm,denf,ff,xn,gis,ierr)
      tc00=tc00+step/RBIMOD(e)
      if(ierr.eq.0) dlen=dlen+step
      if(iswt.eq.1) goto 73
      st1=st2
      st2=st2+step
      goto 72
   73 continue
      tinmod(i)=-1.0d0
      tc00=tc00*R
      if(dlen.le.0.01) goto 6
c
c    compute fresnel zone
c
      else
      CALL FRESNEL(type,type1,delta,x1,x2,xn,IRAY2(i),nm,lidd,nlidd,
     +denf,den1f,den2f,ff,tc00,ierr)
      tinmod(i)=-1.0d0
      if(nlidd.eq.0) goto 6
      endif
c
c compute anisotropy along the Gaussian ray
c
      if(ianiz.ne.0) then
        do j=nm+1,nmAL
          ff(j)=0.0
        enddo
        nlidd1=0
        lidd1(1)=0
        step=step0*drad
        iswt=0
        st1=0.0d0
        st2=step
        dlen=0.0d0
    2   if(st2.ge.delta) then
        st2=delta
        step=st2-st1
        iswt=1
        endif
        firot=(st2+st1)/2.0d0
        CALL RTURN(firot,xn,x1,e)
        call NORM(e,e)
        CALL ESTRACE(lidd1,nlidd1,step,e,nm,nma,ff,xn,gis,ierr)
        if(ierr.eq.0) dlen=dlen+step
        if(iswt.eq.1) goto 3
        st1=st2
        st2=st2+step
        goto 2
    3   continue
      tinmod(i)=-1.0d0
c     tc00=tc00*R
      if(dlen.le.0.01) goto 6
      endif
c
c  evaluate total density and for R1/R2
c
      do 31 j=1,nm
      den1fs(j)=den1fs(j)+den1f(j)
      den2fs(j)=den2fs(j)+den2f(j)
   31 denfs(j)=denfs(j)+denf(j)
      nff=nff+1
      tinmod(i)=tc00
c
c     Matrix creation
c
      w2w=WEIGHT(i)*WEIGHT(i)
      if(.not.ltrace) then
        if(ianiz.ne.0) then
          do  k=1,2*ianiz-1,2
            do kk=1,nlidd1
              lidd(nlidd+(k-1)*nlidd1+kk)=lidd1(kk)+(k-1)*nma+nm
              lidd(nlidd+k*nlidd1+kk)=lidd1(kk)+k*nma+nm
            enddo
          enddo
          nlidd=nlidd+2*ianiz*nlidd1
        endif
        do 4 lii=1,nlidd
          ii=lidd(lii)
          do 4 lj=lii,nlidd
            j=lidd(lj)
            s=ff(ii)*ff(j)*w2w
            s1=f((ii-1)*nmAL+j)+s
            f((ii-1)*nmAL+j)=s1
            if(ii.ne.j) f((j-1)*nmAL+ii)=s1
    4   continue
      endif
        s2sn=t(i)-tc00
        s2s=s2sn*w2w
        do ii=1,nmAL
          fsl(ii)=fsl(ii)+ff(ii)*s2s
        enddo
      resid_i(i)=s2sn
      sigs2s=sigs2s+DBLE(s2sn)*DBLE(s2sn)
      sigw2w=sigw2w+DBLE(s2s)*DBLE(s2s)
      sss=DBLE(R)*delta
      sss=sss*(1.0d0/DBLE(t(i))-1.0d0/tc00)
      sigv2v=sigv2v+sss*sss
c
c In case of Ray Tracing print results
c
      if(ltrace) then
        fff0=TE0(i)
        fff1=TE(i)
        if(lcoord) then
          fff0= ATAN(TAN(const*TE0(i))/GEO)/const
          fff1 = ATAN(TAN(const*TE(i))/GEO)/const
        endif
        write(35,'(i7,4f12.4,2f12.5,i2,3f12.5)') IIII(i),
     *   fff0,FI0(i),fff1,FI(i),delta*6371.0d0/DBLE(T(i)),
     *   WEIGHT(i),IRAY2(i),delta*6371.0d0/tc00,resid_i(i),
     *   SNGL(delta/drad)
      endif
    6 continue
    7 continue
      write(*,*) 'R1 traces:',n1,' R2 Traces:',n2
      write(8,*) 'R1 traces:',n1,' R2 Traces:',n2
c
c store matrix G in a disk file
c
      if((ldresol.or.lresol).and..not.ltrace) CALL WRITE_BIN(nmAL,f,fdres)
      ns2s=nff
      sigs2s=DSQRT(sigs2s/(DBLE(ns2s)-1.0d0))
      sigv2v=DSQRT(sigv2v/(DBLE(ns2s)-1.0d0))
      sigw2w=DSQRT(sigw2w/(DBLE(ns2s)-1.0d0))
      write(*,*)'Number of integrals are: ',ns2s
      write(8,*)'Number of integrals are: ',ns2s
      write(*,2000) sigs2s,sigw2w
      write(8,2000) sigs2s,sigw2w
      write(*,2001) sigv2v
      write(8,2001) sigv2v
 2000 format("SQRT of initial residuals dispersion are: ",f9.3,
     +" sec /",f9.3,"/")
 2001 format("SQ_RT of initial velocities dispersion are: ",f9.5,
     +" km/sec")
      pstr='Integrals '
      CALL PSTIME(pstr)
c
c In case of Ray Tracing mode stop computations
c
      if(ltrace) then
         close(8)
         close(35)
         STOP '(R0098) END of RAY TRACING'
      endif
c
c   Azimutal distribution
c
      azmax=0.0
      nmgis=nma
      if(ianiz.eq.0) nmgis=nm
      do kk=1,nmgis
        s=0.0
        s2=0.0
        gsum=0.0
        do kkk=1,IAZIM
          gise=gis(kkk,kk)
          s=s+gise
          s2=s2+gise*gise
          if(gsum.lt.gise) gsum=gise
        enddo
        dazi(kk)=0.0
        d2azi(kk)=0.0
        if(gsum.gt.0.0) dazi(kk)=s/gsum*180.0/FLOAT(IAZIM)
        if(dazi(kk).gt.azmax) azmax=dazi(kk)
        if(s.gt.4.5) d2azi(kk)=s*s/s2
      enddo
      write(*,*)'MAXIMUM of asimutal parameter is : ',azmax
      write(8,*)' MAXIMUM of asimutal parameter is : ',azmax
c
c   recompute density map for anisotropic part
c
      if(ianiz.ne.0) then
        do 111 i=1,nma
          iou=ioutra(i)
          i1=icrpnta(iou)/1000000
          i3=icrpnta(iou)-i1*1000000
          i2=i3/1000
          i3=i3-i2*1000
          s=DSQRT(dr(i1)*dr(i1)+dr(i2)*dr(i2)+dr(i3)*dr(i3))
          xh=dr(i1)/s
          yh=dr(i2)/s
          zh=dr(i3)/s
          xk=DACOS(zh)
          yk=DATAN2(yh,xh)
cxx   xk=DATAN(DTAN(dpi/2.0d0-xk)/GEO)/drad
          wval(1)=0.0
          CALL EINTPOL   (xk,yk,nm,denfs,wval,1,ierr)
          denss(i)=wval(1)
  111   continue
      endif
c
c Continue  matrix creation. Create lines with smoothing
c
      if(alpha1.lt.0.0) goto 40
        do 130 i=1,nm
          denf(i)=0.0
          do 210 j=1,nm
  210       ff(j)=0.0
          nlidd=0
          CALL EGAUSS(lidd,nlidd,i,nm,ff,denfs,denf(i),
     +                den1fs,den1f(i),den2fs,den2f(i))
          nff=nff+1
          do 229 lii=1,nlidd
            ii=lidd(lii)
            do 229 ljj=lii,nlidd
              jj=lidd(ljj)
              s=ff(ii)*ff(jj)*wexp*wexp
              s1=f((ii-1)*nmAL+jj)+s
              f((ii-1)*nmAL+jj)=s1
              if(ii.ne.jj) f((jj-1)*nmAL+ii)=s1
  229     continue
  130   continue
c
c Anisotropic part matrix calculation
c
      if(ianiz.ne.0) then
        do 30 i=1,nma
          dens(i)=0.0
          do 10 j=nm+1,nmAL
   10       ff(j)=0.0
          nlidd=0
          CALL EGAUSS_2(lidd,nlidd,i,nm,nma,ff,denss,dens(i))
          nff=nff+NAZIP-1
            do 20 lii=1,nlidd
              ii=lidd(lii)/10
              ip=lidd(lii)-ii*10
              do 20 ljj=lii,nlidd
                jj=lidd(ljj)/10
                jp=lidd(ljj)-jj*10
                if(ip.ne.jp) goto 20
                s=ff(ii)*ff(jj)*wexp1*wexp1
                s1=f((ii-1)*nmAL+jj)+s
                f((ii-1)*nmAL+jj)=s1
                if(ii.ne.jj) f((jj-1)*nmAL+ii)=s1
   20     continue
   30   continue
      endif
c
c   Print timestamp
c
   40 continue
      pstr='Gauss     '
      call PSTIME(pstr)
c
c Calculate number of zeroed and nonzeroed matrix elements
c
      iccc=0
      do 50 i=1,(nm+nma*(NAZIP-1))*(nm+nma*(NAZIP-1))
      if(f(i).eq.0.0) iccc=iccc+1
   50 continue
      write(8,*) 'Matrix is done'
      write(*,*) 'Matrix is done'
      write(*,*) 'Matrix elements:',
     +(NRAZ+NRAZA*(NAZIPL-1))*(NRAZ+NRAZA*(NAZIPL-1)),' Zero are:',iccc
      write(8,*) 'Matrix elements:',
     +(NRAZ+NRAZA*(NAZIPL-1))*(NRAZ+NRAZA*(NAZIPL-1)),' Zero are:',iccc
      write(*,*) 'Total number of matrix lines are',nff
      write(8,*) 'Total number of matrix lines are',nff
c
c  Compute max density
c
      wsum=0.0
      wsum1=0.0
      wsum2=0.0
      do 51 kk=1,nm
cxx   ww=denf(kk)/2.0/wexp
      ww=denf(kk)
      ww1=den1f(kk)
      ww2=den2f(kk)
      if(itomo.eq.0) then
        ww=ww/2.0d0/wexp
        denf(kk)=ww
        ww1=ww
      endif
      if(ww.gt.wsum)wsum=ww
      if(ww1.gt.wsum1)wsum1=ww1
      if(ww2.gt.wsum2)wsum2=ww2
   51 continue
      write(*,*)'Maximum intersections/cell are: ',wsum,' R1:',wsum1,' R2:',wsum2
      write(8,*)'Maximum intersections/cell are: ',wsum,' R1:',wsum1,' R2:',wsum2
c
c Regularization procedure:
c adding to diagonal elements regularization factor   
c
      sre=0.0
      sre1=0.0
      do 60 kk=1,nmAL
      if(kk.le.nm) sre =sre +f((kk-1)*nmAL+kk)/nm
      if(kk.gt.nm) sre1=sre1+f((kk-1)*nmAL+kk)/(nmAL-nm)
   60 continue
      if(alpha.gt.0.0) then
      do 70 kk=1,nmAL
      if(kk.le.nm) then
        dm=denf(kk)
        sre_norm=0.0
        if(abs(dm).lt.200.0) sre_norm=EXP(dump1*dm)
      endif
      if(kk.gt.nm) then
        dm=dens(MOD(kk-1-nm,nma)+1)
        sre_norm=0.0
        if(abs(dm).lt.200.0) sre_norm=EXP(dump2*dm)
      endif
      if(kk.le.nm) f((kk-1)*nmAL+kk)=f((kk-1)*nmAL+kk)+sre*alpha*sre_norm
      if(kk.gt.nm) f((kk-1)*nmAL+kk)=f((kk-1)*nmAL+kk)+sre1*alpha*sre_norm
   70 continue
      endif
      write(8,1003)sre
      write(*,1003)sre
      if(nmAL.gt.nm) then
      write(8,1005)sre1
      write(*,1005)sre1
      endif
 1003 format(' Isotropy   Trace/Size F',f15.4)
 1005 format(' Anisotropy Trace/Size F',f15.4)
c
c  Find inverse matrix using SGECO algorithm
c
      CALL SGECO(f,nmAL,nmAL,ipvt,rcond,dens)
      pstr='Inversion '
      CALL PSTIME(pstr)
      write(8,'("Matrix condition = ",e15.6)') rcond
      write(*,'("Matrix condition = ",e15.6)') rcond
      CALL SGESL(f,nmAL,nmAL,ipvt,fsl,0)
c
c  Covariance matrix computation.
c  Compute invers matrix G+-1 and store it to file.
c
      if(ldresol) then 
      open(25,file=fdres1,form='unformatted',status='unknown')
      do i=1,nmAL
        do j=1,nmAL
          dens(j)=0.0
          if(i.eq.j) dens(j)=1.0
        enddo
        CALL SGESL(f,nmAL,nmAL,ipvt,dens,0)
        write(25) (dens(l),l=1,nmAL)
      enddo
      close(25)
c
c  read invers matrix into array f again
c
      CALL READ_BIN(nmAL,f,fdres1)
      pstr='Inv. matrx'
      CALL PSTIME(pstr)
c
c  compute G+-1 * G
c
      CALL MAT_MULT(nmAL,f,fdres,fdres2)
      pstr='G-1 * G   '
      CALL PSTIME(pstr)
c
c  compute G+-1 * G * G+-1
c
      open(25,file=fdres1,form='unformatted',status='unknown')
      do i=1,nmAL
        read(25) (dens(l),l=1,nmAL)
        s=0.0
        do j=1,nmAL
        s=s+dens(j)*f((i-1)*nmAL+j)
        enddo
        denss(i)=s
      enddo
      close(25)
      pstr='Matrix OK.'
      CALL PSTIME(pstr)
      endif
c
c  Resolution matrix computation.
c  Compute invers matrix G+ * G and store it in array f.
c
      if(lresol) then
       if(.not.ldresol) then
        open(25,file=fdres,form='unformatted',status='old')
        open(26,file=fdres1,form='unformatted',status='unknown')
          do j=1,nm
            read(25) (dens(l),l=1,nmAL)
            CALL SGESL(f,nmAL,nmAL,ipvt,dens,0)
            write(26) (dens(l),l=1,nm)
          enddo
        close(25)
        close(26)
        CALL READ_BIN(nm,f,fdres1)
       else
          open(25,file=fdres2,form='unformatted',status='old')
          open(26,file=fdres1,form='unformatted',status='unknown')
        do j=1,nm
          read(25) (dens(l),l=1,nmAL)
          write(26) (dens(l),l=1,nm)
          i=(j-1)*nm
          do kk=1,nm
            f(i+kk)=dens(kk)
          enddo
        enddo
        close(25)
        close(26)
       endif
        do i=1,nm
          iou=ioutr(i)
          i1=icrpnt(iou)/1000000
          i3=icrpnt(iou)-i1*1000000
          i2=i3/1000
          i3=i3-i2*1000
          sxy=DSQRT(dr(i1)*dr(i1)+dr(i2)*dr(i2)+dr(i3)*dr(i3))
          xh=dr(i1)/sxy
          yh=dr(i2)/sxy
          zh=dr(i3)/sxy
          xk=DACOS(zh)
          yk=DATAN2(yh,xh)/drad
          xk=DATAN(DTAN(dpi/2.0d0-xk)/DBLE(GEO))/drad
          xk1(i)=xk
          yk1(i)=yk
cxx   write(*,*) i,xk1(i),yk1(i)
        enddo
          CALL PSTIME('Res_matrix')
          if(lpoint) CALL RES_OUTP(nm,xk1,yk1,f)
          CALL RES_ANAL(nm,xk1,yk1,f)
        CALL PSTIME('Res_analys')
      endif
c
c Calculate residuals
c
      CALL RESID(type,type1,sigs2s,sigv2v,sigw2w,step0,nm,nma,fsl)
      write(*,2002) sigs2s,sigw2w
      write(8,2002) sigs2s,sigw2w
      write(*,2003) sigv2v
      write(8,2003) sigv2v
 2002 format("SQRT of final residuals dispersion are: ",f9.3,
     +" sec /",f9.3,"/")
 2003 format("SQ_RT of final velocities dispersion are: ",f9.5,
     +" km/sec")
c*********** Return to home ***********************************
      ihome=0
      if(ihome.eq.1) then
      open(13,file='GIST',status='UNKNOWN')
      do 110 i=1,nm
      iou=ioutr(i)
      i1=icrpnt(iou)/1000000
      i3=icrpnt(iou)-i1*1000000
      i2=i3/1000
      i3=i3-i2*1000
      s=DSQRT(dr(i1)*dr(i1)+dr(i2)*dr(i2)+dr(i3)*dr(i3))
      xh=dr(i1)/s
      yh=dr(i2)/s
      zh=dr(i3)/s
      xk=DACOS(zh)/drad
      yk=DATAN2(yh,xh)
      xk=DATAN(DTAN(dpi/2.0d0-xk)/GEO)/drad
      write(13,'(2f7.1,f10.2,10f7.0)') xk,yk,dazi(i)/180.0,(gis(l,i),l=1,IAZIM)
  110 continue
      close(13)
      NXY=nm
      else
      NXY=0
c
c   open xxx.dr data resolution file
c
      if(ldresol) open(26,file=fname6,status='unknown')
      do 120 k=1,nlat
      if(lcoout) then
         xk=ATAN(GEO*TAN((dlat0+(k-1)*s_lat)*const))
      else
         xk=(dlat0+(k-1)*s_lat)*const
      endif
      xk=pi/2.0-xk
c
c Interolate maps for spherical cooordinates
c
      do 120 j=1,nlon
      NXY=NXY+1
      yk=(dlon0+(j-1)*s_lon)*const
      value1=RBIMOD1(xk,yk)
      wval(1)=0.0
      wval1(1)=0.0
      wval2(1)=0.0
      gval(1)=0.0
      gval2(1)=0.0
      dvalue(1)=0.0
      if(ianiz.eq.0) then
        CALL EINTPOL (xk,yk,nm,dazi,gval,1,ierr)
        CALL EINTPOL (xk,yk,nm,d2azi,gval2,1,ierr)
      else
        CALL EINTPOL_2 (xk,yk,nm,nma,dazi,gval,1,ierr)
        CALL EINTPOL_2 (xk,yk,nm,nma,d2azi,gval2,1,ierr)
      endif
      CALL EINTPOL   (xk,yk,nm,denf,wval,1,ierr)
      CALL EINTPOL   (xk,yk,nm,den1f,wval1,1,ierr)
      CALL EINTPOL   (xk,yk,nm,den2f,wval2,1,ierr)
      CALL EINTPOL   (xk,yk,nm,fsl,value,1,ierr)
      if(ianiz.ne.0) CALL EINTPOL_2 (xk,yk,nm,nma,fsl,value,NAZIP,ierr)
      if(ldresol) then
        CALL EINTPOL   (xk,yk,nm,denss,dvalue,1,ierr)
        if(ianiz.ne.0) CALL EINTPOL_2 (xk,yk,nm,nma,denss,dvalue,NAZIP,ierr)
        vvv=value1/(1.+value(1))
        write(26,*) dlon0+(j-1)*s_lon,dlat0+(k-1)*s_lat,
     *  (dvalue(l)*vvv*vvv/1.0E-7,l=1,NAZIP)
      endif
      if(ierr.eq.1) then
      NXY=NXY-1
      value(1)=0.0
      goto 120
      endif
      CALL DUAZI(xk,yk,duaz)
c     write(*,*)duaz
c
c   Main output
c
      if(ianiz.eq.1) then
        vvv=value1/(1.+value(1))
        rrr=value(2)*vvv*vvv/value1
        rrrr=value(3)*vvv*vvv/value1
        if(istrip.eq.0) then
          value(2)=(rrr*DCOS(2.0d0*duaz)+rrrr*DSIN(2.0d0*duaz))
          value(3)=(-rrr*DSIN(2.0d0*duaz)+rrrr*DCOS(2.0d0*duaz))
          vamp1=SQRT(value(2)*value(2)+value(3)*value(3))
          vph1=(pi/4.0 - ATAN2(value(2),value(3))/2.0)/const
          if(vph1.lt.0.0)vph1=vph1+360.0
          write(10,1004)dlon0+(j-1)*s_lon,dlat0+(k-1)*s_lat,
     *    vvv,value1,value(1),vamp1,vph1,value(2),value(3)
        else
          value(2)=(rrr*DCOS(4.0d0*duaz)+rrrr*DSIN(4.0d0*duaz))
          value(3)=(-rrr*DSIN(4.0d0*duaz)+rrrr*DCOS(4.0d0*duaz))
          vamp1=SQRT(value(2)*value(2)+value(3)*value(3))
          vph1=(pi/8.0 - ATAN2(value(2),value(3))/4.0)/const
          if(vph1.lt.0.0)vph1=vph1+360.0
          write(10,1004)dlon0+(j-1)*s_lon,dlat0+(k-1)*s_lat,
     *         vvv,value1,value(1),0.0,0.0,0.0,0.0,
     *         vamp1,vph1,value(2),value(3)
        endif
cxx     vph1=ATAN2(rrrr,rrr)/2.0/const
cxx     if(vph1.lt.0.0)vph1=vph1+360.0
      else if(ianiz.eq.2) then
c*************************************************
        vvv=value1/(1.+value(1))
        rrr=value(2)*vvv*vvv/value1
        rrrr=value(3)*vvv*vvv/value1
        value(2)=(rrr*DCOS(2.0d0*duaz)+rrrr*DSIN(2.0d0*duaz))
        value(3)=(-rrr*DSIN(2.0d0*duaz)+rrrr*DCOS(2.0d0*duaz))
        vamp1=SQRT(value(2)*value(2)+value(3)*value(3))
        vph1=(pi/4.0 - ATAN2(value(2),value(3))/2.0)/const
        if(vph1.lt.0.0)vph1=vph1+360.0
c*************************************************
        rrr4=value(4)*vvv*vvv/value1
        rrrr4=value(5)*vvv*vvv/value1
        value(4)=rrr4*DCOS(4.0d0*duaz)+rrrr4*DSIN(4.0d0*duaz)
        value(5)=-rrr4*DSIN(4.0d0*duaz)+rrrr4*DCOS(4.0d0*duaz)
        vamp2=SQRT(value(4)*value(4)+value(5)*value(5))
        vph2=(pi/8.0 - ATAN2(value(4),value(5))/4.0)/const
        if(vph1.lt.0.0)vph1=vph1+360.0
        write(10,1004)dlon0+(j-1)*s_lon,dlat0+(k-1)*s_lat,
     *  vvv,value1,value(1),vamp1,vph1,value(2),value(3),
     *  vamp2,vph2,value(4),value(5)
      else
        uuu=dlon0+(j-1)*s_lon
cxx     if(uuu.gt.180.) uuu=uuu - 360.
        write(10,1004) uuu,dlat0+(k-1)*s_lat,value1/(1.+value(1))
      endif
 1004 format(4f12.4,9f12.6)
 1006 format(5f12.4)
      if(lres) then
      write(9,1006)dlon0+(j-1)*s_lon,dlat0+(k-1)*s_lat,wval(1),wval1(1),wval2(1)
      write(11,1004)dlon0+(j-1)*s_lon,dlat0+(k-1)*s_lat,gval2(1),gval(1)
      endif
 120  continue
c
c    close temorary  file
c
      if(ldresol) then
        close(26)
        l=lnblnk(fdres)
        if(lpoint) then
          cmd='/bin/rm '//fdres(1:l)//' '//fdres2(1:l)
        else
          cmd='/bin/rm '//fdres(1:l)//' '//fdres1(1:l)//' '//fdres2(1:l)
        endif
        CALL SYSTEM(cmd)
      endif
      if((.not.ldresol).and.lresol) then
        l=lnblnk(fdres)
        if(lpoint) then
          cmd='/bin/rm '//fdres(1:l)
        else
          cmd='/bin/rm '//fdres(1:l)//' '//fdres1(1:l)
        endif
        CALL SYSTEM(cmd)
      endif
      if(lresol.and.lpoint) then
        ll=lnblnk(fdres1)
        l=lnblnk(fname6)
        cmd='mv '//fdres1(1:ll)//' '//fname6(1:l)//'2'
        CALL SYSTEM(cmd)
      endif
c
c next statement started from ihome
c      
      endif
      pstr='Velocities'
      CALL PSTIME(pstr)
c
c   End. Return to main program
c
      close(10)
      close(11)
      return
      end
c****************************************************************
c compute apossteriori residuals
c****************************************************************
      SUBROUTINE RESID(type,type1,sigs2s,sigv2v,sigw2w,stepi,nmm,
     +nmma, fsl)
      IMPLICIT NONE
      include "tomo.h"
      include "line.h"
c TYPE definitions +++++++++++++++++++++++++++++++++++++++++++++++++++
      character*4 type,type1
      integer*4 nmm, nmma
      real*4    fsl(NRAZ+NRAZA*(NAZIPL-1))
      real*4    fsum,sum,s2s,w2w,fff0,fff1
      integer*4 i,ierr,iswt,nff,n_p
      real*8    sigs2s,sigv2v,sigw2w,sss,fsum1
      real*8    x1(3),x2(3),xn(3),e(3),delta,step,stepi,dlen,st1,
     +          st2,firot
c END TYPE definitions ++++++++++++++++++++++++++++++++++++++++++++++++
      nff=0
      sigs2s=0.0d0
      sigv2v=0.0d0
      sigw2w=0.0d0
      do 3 i=1,n
c
c Get normal xn to ray plane & epicentral distance
c
      CALL TO_CART(TE0(i),FI0(i),TE(i),FI(i),IRAY2(i),x1,x2,xn,delta)
      fsum=0.0
      if(itomo.eq.0) then
c
c compute residuals for Gaussian ray
c
      step=stepi*drad
      iswt=0
      st1=0.
      st2=step
      dlen=0.
   81 if(st2.ge.delta) then
      st2=delta
      step=st2-st1
      iswt=1
      endif
      firot=(st2+st1)/2.
      CALL RTURN(firot,xn,x1,e)
      call NORM(e,e)
      CALL EINTEGRN(step,e,xn,nmm,sum,fsl,ierr)
      if(ierr.eq.0) dlen=dlen+step
      if(ierr.eq.0) fsum=fsum+sum
      if(iswt.eq.1) goto 82
      st1=st2
      st2=st2+step
      goto 81
   82 continue
      if(tinmod(i).lt.0.0d0) goto 3
      if(dlen.le.0.01) goto 3
      else 
c
c compute residuals for fresnel zone
c
      fsum1=0.0d0
      CALL FR_RES(type,type1,delta,x1,x2,xn,IRAY2(i),nmm,fsum1,fsl,n_p,ierr)
      if(tinmod(i).lt.0.0d0) goto 3
      if(n_p.eq.0) goto 3
      fsum=SNGL(fsum1)+fsum
      endif
c
c compute residuals for anisotropic part
c
      if(ianiz.ne.0) then
        step=stepi*drad
        iswt=0
        st1=0.0d0
        st2=step
        dlen=0.0d0
    1   if(st2.ge.delta) then
        st2=delta
        step=st2-st1
        iswt=1
        endif
        firot=(st2+st1)/2.0d0
        CALL RTURN(firot,xn,x1,e)
        call NORM(e,e)
        CALL EINTEGR_2(step,e,xn,nmm,nmma,sum,fsl,ierr)
        if(ierr.eq.0) dlen=dlen+step
        if(ierr.eq.0) fsum=fsum+sum
        if(iswt.eq.1) goto 2
        st1=st2
        st2=st2+step
        goto 1
    2   continue
      endif
      nff=nff+1
      w2w=WEIGHT(i)*WEIGHT(i)
      s2s=fsum-(t(i)-tinmod(i))
c
c  output residuals
c
      if(lresid) then
        fff0=TE0(i)
        fff1=TE(i)
        if(lcoord) then
          fff0= ATAN(TAN(const*TE0(i))/GEO)/const
          fff1 = ATAN(TAN(const*TE(i))/GEO)/const
        endif
        write(35,'(i7,4f12.4,2f12.5,i2,3f12.5)') IIII(i),
     *   fff0,FI0(i),fff1,FI(i),delta*6371.0d0/DBLE(T(i)),
     *   WEIGHT(i),IRAY2(i),0.0-s2s,resid_i(i),SNGL(delta/drad)
      endif
      sigs2s=sigs2s+DBLE(s2s*s2s)
      sigw2w=sigw2w+DBLE(s2s*s2s*w2w)
      sss=DBLE(R)*delta
      sss=sss*(1.0d0/DBLE(t(i))-1.0d0/(DBLE(fsum)+tinmod(i)))
      sigv2v=sigv2v+sss*sss
    3 continue
      if(lresid) close(35)
      sigs2s=DSQRT(sigs2s/(DBLE(nff)-1.0d0))
      sigv2v=DSQRT(sigv2v/(DBLE(nff)-1.0d0))
      sigw2w=DSQRT(sigw2w/(DBLE(nff)-1.0d0))
      return
      end
c****************************************************************
c  get velocity from reference model by Cartesian coordinates
c****************************************************************
      REAL FUNCTION RBIMOD(e)
      IMPLICIT NONE
      include "tomo.h"
      include "line.h"
      integer*4 i,j
      real*8    dff,dfl,r0,r1,r2,r3,di,dj,e(3)

      if(.not. lmodel) goto 1
      dfl=DATAN2(e(2),e(1))/drad
      if(dfl.lt.0.d0)dfl=dfl+360.d0
      dff=e(3)
      if(DABS(dff).gt.1.d0) dff=DSIGN(1.d0,e(3))
      dff=DACOS(dff)/drad
      if(lcoord) dff = 90.0d0-datan(dtan((90.0d0-dff)*drad)/GEO)/drad
      i=dfl+1.d0
      j=dff+1.d0
      if(i.lt.1)i=1
      if(i.gt.359)i=359
      if(j.lt.1)j=1
      if(j.gt.180)j=180
      di=dfl-i+1
      dj=dff-j+1
      r0=f2(j,i)
      r1=f2(j+1,i)-f2(j,i)
      r2=f2(j,i+1)-f2(j,i)
      r3=f2(j,i)+f2(j+1,i+1)-f2(j,i+1)-f2(j+1,i)
      RBIMOD=r0+r1*dj+r2*di+r3*di*dj
      return
    1 RBIMOD=c00
      return
      end
c****************************************************************
c  get velocity from reference model by spherical coordinates
c****************************************************************
      REAL FUNCTION RBIMOD1(xk,yk)
      IMPLICIT NONE
      real*8 dff,dfl,r0,r1,r2,r3,di,dj,xk,yk
      include "tomo.h"
      include "line.h"
      integer*4 i,j
      if(.not. lmodel) goto 1
      dfl=yk/drad
      if(dfl.lt.0.d0)dfl=dfl+360.d0
      dff=xk/drad
      if(lcoord) dff = 90.0d0-datan(dtan((90.0d0-dff)*drad)/GEO)/drad
      i=dfl+1.d0
      j=dff+1.d0
      if(i.lt.1)i=1
      if(i.gt.359)i=359
      if(j.lt.1)j=1
      if(j.gt.180)j=180
      di=dfl-i+1
      dj=dff-j+1
      r0=f2(j,i)
      r1=f2(j+1,i)-f2(j,i)
      r2=f2(j,i+1)-f2(j,i)
      r3=f2(j,i)+f2(j+1,i+1)-f2(j,i+1)-f2(j+1,i)
      RBIMOD1=r0+r1*dj+r2*di+r3*di*dj
      return
    1 RBIMOD1=c00
      return
      end
c******************************************************
c The same as RBIMOD but with double precision
c******************************************************
      REAL*8 FUNCTION RBIMOD_8(e)
      IMPLICIT NONE
      include "tomo.h"
      include "line.h"
      integer*4 i,j
      real*8 dff,dfl,r0,r1,r2,r3,di,dj,e(3)
      if(.not. lmodel) goto 1
      dfl=DATAN2(e(2),e(1))/drad
      if(dfl.lt.0.d0)dfl=dfl+360.d0
      dff=e(3)
      if(DABS(dff).gt.1.d0) dff=DSIGN(1.d0,e(3))
      dff=DACOS(dff)/drad
      i=dfl+1.d0
      j=dff+1.d0
      if(i.lt.1)i=1
      if(i.gt.359)i=359
      if(j.lt.1)j=1
      if(j.gt.180)j=180
      di=dfl-i+1
      dj=dff-j+1
      r0=f2(j,i)
      r1=f2(j+1,i)-f2(j,i)
      r2=f2(j,i+1)-f2(j,i)
      r3=f2(j,i)+f2(j+1,i+1)-f2(j,i+1)-f2(j+1,i)
      RBIMOD_8=r0+r1*dj+r2*di+r3*di*dj
      return
    1 RBIMOD_8= dble(c00)
      return
      end
c****************************************************************
c Output current time relative to the start program time
c****************************************************************
      SUBROUTINE PSTIME(str)
      IMPLICIT NONE
      save
c TYPE definitions ++++++++++++++++++++++++++++++++++++++
      character*10 str
      character*24 stime,ctime
      integer*4 i,ibeg,ih,im,indt,is,itm,itime,time
c END TYPE definitions ++++++++++++++++++++++++++++++++++
      data indt/0/
      i=time()
      stime=ctime(i)
      if(indt.eq.0) then
      indt=1
      ibeg=i
      endif
      itm=i-ibeg
      itime=itm
      ibeg=i
      ih=itm/3600
      itm=itm-ih*3600
      im=itm/60
      is=itm-im*60
      write(*,'(a10," - <",a24,"> Duration: ",i2,":",i2,":",i2,i6)') str,stime,ih,im,is,itime
      write(8,'(a10," - <",a24,">, Duration: ",i2,":",i2,":",i2,i6)') str,stime,ih,im,is,itime
      return
      end
