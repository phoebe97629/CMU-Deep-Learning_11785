import numpy as np


def GreedySearch(SymbolSets, y_probs):
    """Greedy Search.

    Input
    -----
    SymbolSets: list
                all the symbols (the vocabulary without blank)

    y_probs: (# of symbols + 1, Seq_length, batch_size)
            Your batch size for part 1 will remain 1, but if you plan to use your
            implementation for part 2 you need to incorporate batch_size.

    Returns
    ------
    forward_path: str
                the corresponding compressed symbol sequence i.e. without blanks
                or repeated symbols.

    forward_prob: scalar (float)
                the forward probability of the greedy path

    """
    # Follow the pseudocode from lecture to complete greedy search :-)
    #choose max prob
    symbol, seq_len, batch = y_probs.shape
    forward_path = []
    forward_prob = 1
    #since y_probs = symbols +1
    SymbolSets = [''] + SymbolSets
    last_index = 0
    for i in range(seq_len):
        index = int(np.argmax(y_probs[:, i, :], axis=0))
        forward_prob *= y_probs[index, i]
        if index != 0:
            #repeated error
            if bool(forward_path) == False or SymbolSets[index] != forward_path[-1] or last_index == 0:
                forward_path.append(SymbolSets[index])
        last_index = index

    combined_forward_path = ''
    for i in forward_path:
        combined_forward_path += i

    forward_path = combined_forward_path
    return (forward_path, forward_prob)
    #raise NotImplementedError


##############################################################################


def BeamSearch(SymbolSets, y_probs, BeamWidth):
    """Beam Search.

    Input
    -----
    SymbolSets: list
                all the symbols (the vocabulary without blank)

    y_probs: (# of symbols + 1, Seq_length, batch_size)
            Your batch size for part 1 will remain 1, but if you plan to use your
            implementation for part 2 you need to incorporate batch_size.

    BeamWidth: int
                Width of the beam.

    Return
    ------
    bestPath: str
            the symbol sequence with the best path score (forward probability)

    mergedPathScores: dictionary
                        all the final merged paths with their scores.

    """
    # Follow the pseudocode from lecture to complete beam search :-)


    def InitializePaths(SymbolSet, y):
        InitialBlankPathScore = {}
        InitialPathScore = {}
    # First push the blank into a path-ending-with-blank stack. No symbol has been invoked yet
        path = ''
        InitialBlankPathScore[path] = y[0, :]  # Score of blank at t=1
        InitialPathsWithFinalBlank = [path]
    # Push rest of the symbols into a path-ending-with-symbol stack
        InitialPathsWithFinalSymbol = []
        # This is the entire symbol set, without the blank
        for c in range(len(SymbolSet)):
            path = SymbolSet[c]
            InitialPathScore[path] = y[c+1,:]
            InitialPathsWithFinalSymbol.append(path)

        return InitialPathsWithFinalBlank, InitialPathsWithFinalSymbol, InitialBlankPathScore, InitialPathScore

    def ExtendWithBlank(PathsWithTerminalBlank, PathsWithTerminalSymbol, y, PathScore, BlankPathScore):
        UpdatedPathsWithTerminalBlank = []
        UpdatedBlankPathScore = {}
        # First work on paths with terminal blanks
        # (This represents transitions along horizontal trellis edges for blanks)
        for path in PathsWithTerminalBlank:
        # Repeating a blank doesn’t change the symbol sequence
            UpdatedPathsWithTerminalBlank.append(path)  # Set addition
            UpdatedBlankPathScore[path] = BlankPathScore[path] * y[0]

        # Then extend paths with terminal symbols by blanks
        for path in PathsWithTerminalSymbol:
        # If there is already an equivalent string in UpdatesPathsWithTerminalBlank
        # simply add the score. If not create a new entry
            if path in UpdatedPathsWithTerminalBlank:
                UpdatedBlankPathScore[path] += PathScore[path] * y[0]
            else:
                UpdatedPathsWithTerminalBlank.append(path)  # Set addition
                UpdatedBlankPathScore[path] = PathScore[path] * y[0]

        return UpdatedPathsWithTerminalBlank,UpdatedBlankPathScore

    def ExtendWithSymbol(PathsWithTerminalBlank, PathsWithTerminalSymbol, SymbolSet, y, PathScore, BlankPathScore):
        UpdatedPathsWithTerminalSymbol = []
        UpdatedPathScore = {}
        # First extend the paths terminating in blanks. This will always create a new sequence
        for path in PathsWithTerminalBlank:
            for c in range(len(SymbolSet)):  # SymbolSet does not include blanks
                newpath = path + SymbolSet[c]  # Concatenation
                UpdatedPathsWithTerminalSymbol.append(newpath)  # Set addition
                UpdatedPathScore[newpath] = BlankPathScore[path] * y[c+1]

        # Next work on paths with terminal symbols
        for path in PathsWithTerminalSymbol:
        # Extend the path with every symbol other than blank
            for c in range(len(SymbolSet)):
                # SymbolSet does not include blanks
                newpath = path if SymbolSet[c] == path[-1] else path + SymbolSet[c] # Horizontal transitions don’t extend the sequence
                if newpath in UpdatedPathsWithTerminalSymbol:  # Already in list, merge paths
                    UpdatedPathScore[newpath] += PathScore[path] * y[c+1]
                else: # Create new path
                    UpdatedPathsWithTerminalSymbol.append(newpath)  # Set addition
                    UpdatedPathScore[newpath] = PathScore[path] * y[c+1]

        return UpdatedPathsWithTerminalSymbol, UpdatedPathScore


    def Prune(PathsWithTerminalBlank, PathsWithTerminalSymbol, BlankPathScore, PathScore, BeamWidth):
        PrunedBlankPathScore = {}
        PrunedPathScore = {}
        # First gather all the relevant scores
        scorelist  = []
        for p in PathsWithTerminalBlank:
            scorelist.append(BlankPathScore[p])
        scorelist += [PathScore[p] for p in PathsWithTerminalSymbol]

        # Sort and find cutoff score that retains exactly BeamWidth paths
        scorelist.sort(reverse=True)  # In decreasing order

        cutoff = scorelist[BeamWidth - 1] if BeamWidth < len(scorelist) else scorelist[-1]
        PrunedPathsWithTerminalBlank = []
        for p in PathsWithTerminalBlank:
            if BlankPathScore[p] >= cutoff:
                PrunedPathsWithTerminalBlank.append(p)# Set addition
                PrunedBlankPathScore[p] = BlankPathScore[p]

        PrunedPathsWithTerminalSymbol = []
        for p in PathsWithTerminalSymbol:
            if PathScore[p] >= cutoff:
                PrunedPathsWithTerminalSymbol.append(p)  # Set addition
                PrunedPathScore[p] = PathScore[p]

        return PrunedPathsWithTerminalBlank, PrunedPathsWithTerminalSymbol, PrunedBlankPathScore, PrunedPathScore


    def MergeIdenticalPaths(PathsWithTerminalBlank, BlankPathScore,PathsWithTerminalSymbol, PathScore):
    # All paths with terminal symbols will remain
        MergedPaths = PathsWithTerminalSymbol
        FinalPathScore = PathScore
        # Paths with terminal blanks will contribute scores to existing identical paths from # PathsWithTerminalSymbol if present, or be included in the final set, otherwise
        for p in PathsWithTerminalBlank:
            if p in MergedPaths:
                FinalPathScore[p] += BlankPathScore[p]
            else:
                MergedPaths.append(p)  # Set addition
                FinalPathScore[p] = BlankPathScore[p]

        return MergedPaths, FinalPathScore

    #GlobalPathScore = [], BlankPathScore = []
    # First time instant: Initialize paths with each of the symbols,
    # including blank, using score at time t=1
    NewPathsWithTerminalBlank, NewPathsWithTerminalSymbol, NewBlankPathScore, NewPathScore = InitializePaths(SymbolSets, y_probs[:, 0])

    # Subsequent time steps
    for t in range(1, y_probs.shape[1]):
    # Prune the collection down to the BeamWidth
        PathsWithTerminalBlank, PathsWithTerminalSymbol, BlankPathScore, PathScore = Prune(NewPathsWithTerminalBlank, NewPathsWithTerminalSymbol,
                                                                                           NewBlankPathScore, NewPathScore, BeamWidth)

    # First extend paths by a blank
        NewPathsWithTerminalBlank, NewBlankPathScore = ExtendWithBlank(PathsWithTerminalBlank,
                                                                       PathsWithTerminalSymbol, y_probs[:, t], PathScore, BlankPathScore)

    # Next extend paths by a symbol
        NewPathsWithTerminalSymbol, NewPathScore = ExtendWithSymbol(PathsWithTerminalBlank,
                                                                    PathsWithTerminalSymbol, SymbolSets, y_probs[:, t], PathScore, BlankPathScore)

    MergedPaths, FinalPathScore = MergeIdenticalPaths(NewPathsWithTerminalBlank, NewBlankPathScore, NewPathsWithTerminalSymbol, NewPathScore)
    # Pick best path
    BestPath = max(FinalPathScore, key = FinalPathScore.get)  # Find the path with the best score


    return (BestPath, FinalPathScore)

#raise NotImplementedError
