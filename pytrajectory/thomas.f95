FUNCTION THOMAS(ain,b,cin,d) RESULT(x)
    REAL(KIND=8), DIMENSION(:), INTENT(IN) :: ain, b, cin, d
    REAL(KIND=8), DIMENSION(SIZE(d)) :: x
    
    INTEGER :: n
    REAL(KIND=8), DIMENSION(:), ALLOCATABLE :: a, c, ctmp, dtmp
    
    n = SIZE(d)
        
    IF( .NOT. (SIZE(ain) .EQ. SIZE(cin))) THEN
        WRITE(*,*) "ERROR: dimension mismatch (a,c)"
        RETURN
    END IF
    
    IF( SIZE(ain) .EQ. SIZE(b) ) THEN
        IF( .NOT. (ain(1) .EQ. 0.0 .AND. cin(SIZE(cin)) .EQ. 0.0) ) THEN
            WRITE(*,*) "ERROR:"
            RETURN
        ELSE
            a = ain
            c = cin
        END IF
    ELSE
        IF( .NOT. (SIZE(ain) .EQ. SIZE(b)-1)) THEN
            WRITE(*,*) "ERROR:"
            RETURN
        ELSE
            ALLOCATE(a(n))
            a(1) = 0.0_8
            a(2:n) = ain
            
            ALLOCATE(c(n))
            c(1:n-1) = cin
            c(n) = 0.0_8
        END IF
    END IF
    
    ! Forward elimination
    ALLOCATE(ctmp(n-1))
    ALLOCATE(dtmp(n))
    
    ctmp(1) = c(1) / b(1)
    dtmp(1) = d(1) / b(1)
    DO i = 2, n-1
        ctmp(i) = c(i) / (b(i) - ctmp(i-1)*a(i))
        dtmp(i) = (d(i) - dtmp(i-1)*a(i)) / (b(i) - ctmp(i-1)*a(i))
    END DO
    dtmp(n) = (d(n) - dtmp(n-1)*a(n)) / (b(n) - ctmp(n-1)*a(n))
    
    ! Backward substitution
    x(n) = dtmp(n)
    DO i = n-1, 1, -1
        x(i) = dtmp(i) - ctmp(i)*x(i+1)
    END DO
    
    ! Free memory
    DEALLOCATE(a, c, ctmp, dtmp)
END FUNCTION THOMAS