SUBROUTINE lm(F, DF, x0, sol)
    IMPLICIT NONE
    
    EXTERNAL F, DF
    REAL(KIND=8), DIMENSION(:), INTENT(IN) :: x0
    REAL(KIND=8), DIMENSION(:), INTENT(OUT) :: sol
    
    REAL, PARAMETER :: b0=0.2, b1=0.2, mu=0.1
    
    
END SUBROUTINE lm
