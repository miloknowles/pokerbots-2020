ranks = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']

if __name__ == "__main__":
  combs = []
  for i in range(len(ranks)):
    for j in range(len(ranks)):
      if i <= j:
        combs.append(ranks[i] + ranks[j] + 'o')
      else:
        combs.append(ranks[i] + ranks[j] + 's')

  print(len(combs))
  with open("./opponent_clusters.txt", "w") as f:
    for c in combs:
      f.write(c + "\n")
