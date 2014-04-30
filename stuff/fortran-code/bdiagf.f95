FUNCTION bdiagf(M, bshape) RESULT Mb
    IMPLICIT NONE
    
    REAL(KIND=8), INTENT(IN), DIMENSION(:,:) :: M
    REAL(KIND=8), INTENT(IN), DIMENSION(2) :: bshape
    INTEGER :: nrow, ncol, Mrow, Mcol
    REAL(KIND=8), INTENT(OUT), DIMENSION(:,:), ALLOCATABLE :: Mb
    
    nrow = bshape(1)
    ncol = bshape(2)
    
    Mrow = SIZE(M,1)
    Mcol = SIZE(M,2)
    
    n = Mrow / nrow
    
    ALLOCATE(Mb, (nrow, ncol))
    Mb = 0
    
    IF Mcol .NE. ncol .OR. MOD(Mrow, nrow) .NE. 0 THEN
        Mb = M
    ELSE
        DO i = 1, n
            Mb(i*nrow:(i+1)*nrow, i*ncol:(i+1)*ncol) = M(i*nrow:(i+1)*nrow, :)
        END DO
    END IF
    
END FUNCTION bdiagf
