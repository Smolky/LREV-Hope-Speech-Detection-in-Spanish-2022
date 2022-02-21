class PrettyPrintConfussionMatrix ():
    """
    pretty print for confusion matrixes
    @link https://gist.github.com/zachguo/10296432
    """

    def print (self, cm, labels, hide_zeroes = False, hide_diagonal = False, hide_threshold = None):
        
        # To ensure labels are strings
        if not all (isinstance(e, (str)) for e in labels):
            labels = [str(label) for label in labels]
        
        columnwidth = max([len(x) for x in labels] + [5])  # 5 is value length
        empty_cell = " " * columnwidth
        
        # Begin CHANGES
        fst_empty_cell = (columnwidth-3) // 2 * " " + "t/p" + (columnwidth - 3) // 2 * " "
        
        if len(fst_empty_cell) < len (empty_cell):
            fst_empty_cell = " " * (len (empty_cell) - len(fst_empty_cell)) + fst_empty_cell
        # Print header
        print("    " + fst_empty_cell, end=" ")
        # End CHANGES
        
        for label in labels:
            print ("%{0}s".format (columnwidth) % label, end=" ")
            
        print ()
        # Print rows
        for i, label1 in enumerate(labels):
            print ("    %{0}s".format(columnwidth) % label1, end=" ")
            for j in range (len (labels)):
                cell = "%{0}.2f".format (columnwidth) % cm[i, j]
                if hide_zeroes:
                    cell = cell if float (cm[i, j]) != 0 else empty_cell
                if hide_diagonal:
                    cell = cell if i != j else empty_cell
                if hide_threshold:
                    cell = cell if cm[i, j] > hide_threshold else empty_cell
                print (cell, end=" ")
            print ()