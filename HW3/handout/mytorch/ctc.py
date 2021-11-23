import numpy as np


class CTC(object):
    """CTC class."""

    def __init__(self, BLANK=0):
        """Initialize instance variables.

        Argument
        --------
        blank: (int, optional)
                blank label index. Default 0.

        """
        self.BLANK = BLANK

    def targetWithBlank(self, target):
        """Extend target sequence with blank.

        Input
        -----
        target: (np.array, dim = (target_len,))
                target output
        ex: [B,IY,IY,F]

        Return
        ------
        extSymbols: (np.array, dim = (2 * target_len + 1,))
                    extended target sequence with blanks
        ex: [-,B,-,IY,-,IY,-,F,-]
        skipConnect: (np.array, dim = (2 * target_len + 1,))
                    skip connections
        ex: [0,0,0,1,0,0,0,1,0]

        """
        extSymbols = []
        skipConnect = []

        # -------------------------------------------->

        # Your Code goes here
        for i in range(len(target)):
            extSymbols.append(self.BLANK)
            extSymbols.append(target[i])
            skipConnect.append(0)
            if i > 0 and target[i] != target[i-1]:
                skipConnect.append(1)
            else:
                skipConnect.append(0)
        extSymbols.append(self.BLANK)
        skipConnect.append(0)

        # <---------------------------------------------

        return extSymbols, skipConnect

    def forwardProb(self, logits, extSymbols, skipConnect):
        """Compute forward probabilities.

        Input
        -----
        logits: (np.array, dim = (input_len, len(Symbols)))
                predict (log) probabilities

                To get a certain symbol i's logit as a certain time stamp t:
                p(t,s(i)) = logits[t,extSymbols[i]]

        extSymbols: (np.array, dim = (2 * target_len + 1,))
                    extended label sequence with blanks

        skipConnect: (np.array, dim = (2 * target_len + 1,))
                    skip connections

        Return
        ------
        alpha: (np.array, dim = (input_len, 2 * target_len + 1))
                forward probabilities

        """
        S, T = len(extSymbols), len(logits)
        #map
        alpha = np.zeros(shape=(T, S))

        # -------------------------------------------->

        # Your Code goes here
        alpha[0,0] = logits[0, extSymbols[0]]
        alpha[0,1] = logits[0, extSymbols[1]]

        for t in range(1,T):
            alpha[t, 0] = alpha[t-1,0]*logits[t, extSymbols[0]]
            for i in range(1, S):
                alpha[t, i] = alpha[t-1, i-1] + alpha[t-1, i]
                if skipConnect[i]:
                    alpha[t, i] += alpha[t-1, i-2]
                alpha[t,i] *= logits[t, extSymbols[i]]
        #raise NotImplementedError
        # <---------------------------------------------

        return alpha

    def backwardProb(self, logits, extSymbols, skipConnect):
        """Compute backward probabilities.

        Input
        -----

        logits: (np.array, dim = (input_len, len(Symbols)))
                predict (log) probabilities

                To get a certain symbol i's logit as a certain time stamp t:
                p(t,s(i)) = logits[t,extSymbols[i]]

        extSymbols: (np.array, dim = (2 * target_len + 1,))
                    extended label sequence with blanks

        skipConnect: (np.array, dim = (2 * target_len + 1,))
                    skip connections

        Return
        ------
        beta: (np.array, dim = (input_len, 2 * target_len + 1))
                backward probabilities

        """
        S, T = len(extSymbols), len(logits)
        beta = np.zeros(shape=(T, S))

        # -------------------------------------------->

        # Your Code goes here
        beta[-1, -1] = 1
        beta[-1, -2] = 1
        for t in range(T-2, -1, -1):
            beta[t, -1] = beta[t+1, -1]*logits[t+1, extSymbols[-1]]
            for i in range(S-2, -1, -1):
                beta[t,i] = beta[t+1,i]*logits[t+1, extSymbols[i]] + beta[t+1, i+1]*logits[t+1, extSymbols[i+1]]
                if (i<S-3) and skipConnect[i+2]:
                    beta[t, i] += beta[t+1, i+2]*logits[t+1, extSymbols[i+2]]
        #raise NotImplementedError
        # <---------------------------------------------

        return beta

    def postProb(self, alpha, beta):
        """Compute posterior probabilities.

        Input
        -----
        alpha: (np.array, dim = (input_len, 2 * target_len + 1))
                forward probability

        beta: (np.array, dim = (input_len, 2 * target_len + 1))
                backward probability

        Return
        ------
        gamma: (np.array, dim = (input_len, 2 * target_len + 1))
                posterior probability

        """
        [T, S] = alpha.shape
        gamma = np.zeros(shape=(T, S))

        # -------------------------------------------->

        # Your Code goes here
        gamma = alpha*beta
        gamma /= np.sum(gamma, axis=1).reshape((-1, 1))

        #raise NotImplementedError
        # <---------------------------------------------

        return gamma
