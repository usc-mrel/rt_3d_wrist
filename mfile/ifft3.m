function in = ifft3(in)
in = ifft(in,[],1);
in = ifft(in,[],2);
in = ifft(in,[],3);
end