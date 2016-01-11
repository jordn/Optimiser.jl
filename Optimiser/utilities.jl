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
