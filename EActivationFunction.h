#ifndef EACTIVATIONFUNCTION_H_INCLUDED
#define EACTIVATIONFUNCTION_H_INCLUDED

enum EActivationFunction
{
    LINEAR, // Identity
    HEAVISIDE, // Binary step
    LOGISTIC,
    SOFTMAX,
    TANH, // Hyperbolic tangent
    TANH01, // Hyperbolic tangent but limited within the range of 0 and 1
    RECTIFIED_LINEAR_UNIT,
    ARCTAN,
    ARCTAN01, // arctan 0 to 1
    SYMMETRICAL_HARD_LIMIT,
    SINUSOID, // sin
    SINUSOID01, // sin 0 to 1
    GAUSSIAN, // A Radial Basis Function in RBF networks
    NOT_SPECIFIED
};

#endif // EACTIVATIONFUNCTION_H_INCLUDED
