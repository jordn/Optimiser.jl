""" Zero-pads a vector vec to required length len"""
function pad(vec, len, pad=0)
  if length(vec) == len
    return vec
  elseif length(vec) > len
    error("Vec already too long");
  end
  padded = repmat([pad], len)
  for i in 1:length(vec)
    padded[i] = vec[i]
  end
  return padded
end


# Taken from MATLAB R2014b
colors = [
(0.0000, 0.4470, 0.7410),
(0.8500, 0.3250, 0.0980),
(0.9290, 0.6940, 0.1250),
(0.4940, 0.1840, 0.5560),
(0.4660, 0.6740, 0.1880),
(0.3010, 0.7450, 0.9330),
(0.6350, 0.0780, 0.1840)
]
